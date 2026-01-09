# cable_data_loader.py
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.spatial.distance import cdist

def build_vocab(label_dir):
    """扫描所有label json文件，构建一个标签词汇表"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    all_labels = set()
    
    for file_name in os.listdir(label_dir):
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(label_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for shape in data.get('shapes', []):
                all_labels.add(shape['label'])
    
    for label in sorted(list(all_labels)):
        if label not in vocab:
            vocab[label] = len(vocab)
            
    return vocab

def _get_obb_crop(image, points, crop_size=(64, 64)):
    """
    从OBB点获取一个AABB裁剪。
    image: PIL Image
    points: OBB的[[x1, y1], [x2, y2], ...]
    """
    points_np = np.array(points)
    xmin = np.min(points_np[:, 0])
    ymin = np.min(points_np[:, 1])
    xmax = np.max(points_np[:, 0])
    ymax = np.max(points_np[:, 1])
    
    # 防止坐标越界
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image.width, xmax)
    ymax = min(image.height, ymax)

    crop = image.crop((xmin, ymin, xmax, ymax))
    crop = crop.resize(crop_size, Image.BILINEAR)
    return crop

def _find_closest_node(obb_centers, point):
    """找到离给定点最近的OBB中心索引"""
    if not obb_centers:
        return -1
    distances = cdist(np.array([point]), np.array(obb_centers))
    return np.argmin(distances)

class CableDataset(Dataset):
    def __init__(self, image_dir, label_dir, line_dir, file_list, vocab, img_transform):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.line_dir = line_dir
        self.file_list = [f.replace('.jpg', '').replace('.json', '') for f in file_list]
        self.vocab = vocab
        self.img_transform = img_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx]
        img_path = os.path.join(self.image_dir, base_name + '.jpg')
        label_path = os.path.join(self.label_dir, base_name + '.json')
        line_path = os.path.join(self.line_dir, base_name + '.json')

        # 1. 加载图像和JSON
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        with open(line_path, 'r', encoding='utf-8') as f:
            line_data = json.load(f)

        nodes = []
        obb_centers = []
        node_labels_for_debug = [] # 用于调试

        # 2. 处理所有OBB（节点）
        for shape in label_data.get('shapes', []):
            points = shape['points']
            center = np.mean(points, axis=0)
            
            # 视觉特征
            crop = _get_obb_crop(image, points)
            crop_tensor = self.img_transform(crop)
            
            # 位置特征 (归一化)
            pos_tensor = torch.tensor([center[0] / img_w, center[1] / img_h], dtype=torch.float)
            
            # 标签特征
            label_str = shape['label']
            label_index = self.vocab.get(label_str, self.vocab['<UNK>'])
            label_tensor = torch.tensor(label_index, dtype=torch.long)
            
            nodes.append({
                'crop': crop_tensor,
                'pos': pos_tensor,
                'label': label_tensor
            })
            obb_centers.append(center)
            node_labels_for_debug.append(label_str)

        num_nodes = len(nodes)
        if num_nodes == 0:
            # 返回空数据，collate_fn会处理
            return {}, torch.empty(0, 0, dtype=torch.float)

        # 3. 构建邻接矩阵 (Ground Truth)
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        
        for line_shape in line_data.get('shapes', []):
            line_points = line_shape['points']
            # 遍历线段 (p1, p2), (p2, p3), ...
            for i in range(len(line_points) - 1):
                p1 = line_points[i]
                p2 = line_points[i+1]
                
                node_idx1 = _find_closest_node(obb_centers, p1)
                node_idx2 = _find_closest_node(obb_centers, p2)
                
                if node_idx1 != -1 and node_idx2 != -1 and node_idx1 != node_idx2:
                    adj_matrix[node_idx1, node_idx2] = 1.0
                    adj_matrix[node_idx2, node_idx1] = 1.0 # 确保对称

        # 4. 组合节点特征
        batch_crops = torch.stack([n['crop'] for n in nodes])
        batch_pos = torch.stack([n['pos'] for n in nodes])
        batch_labels = torch.stack([n['label'] for n in nodes])
        
        node_data = {
            'crops': batch_crops,
            'pos': batch_pos,
            'labels': batch_labels,
            'num_nodes': num_nodes
        }
        
        return node_data, adj_matrix

def collate_fn(batch):
    """
    将不同N的样本填充为(B, max_N, ...)
    """
    batch = [(n, a) for n, a in batch if n] # 过滤掉空样本
    if not batch:
        return {'crops': torch.empty(0), 'pos': torch.empty(0), 'labels': torch.empty(0), 'mask': torch.empty(0)}, torch.empty(0)

    node_data_list, adj_matrices = zip(*batch)
    
    max_nodes = max(item['num_nodes'] for item in node_data_list)
    B = len(batch)
    
    # 假设 crop_tensor 形状为 (C, H, W)
    C, H, W = node_data_list[0]['crops'].shape[1:]
    
    # 初始化填充后的Tensor
    padded_crops = torch.zeros(B, max_nodes, C, H, W, dtype=torch.float)
    padded_pos = torch.zeros(B, max_nodes, 2, dtype=torch.float)
    padded_labels = torch.zeros(B, max_nodes, dtype=torch.long) # 默认为<PAD> (index 0)
    padded_adj = torch.zeros(B, max_nodes, max_nodes, dtype=torch.float)
    mask = torch.zeros(B, max_nodes, dtype=torch.bool) # True为有效数据

    for i, (node_data, adj_matrix) in enumerate(batch):
        n = node_data['num_nodes']
        padded_crops[i, :n] = node_data['crops']
        padded_pos[i, :n] = node_data['pos']
        padded_labels[i, :n] = node_data['labels']
        padded_adj[i, :n, :n] = adj_matrix
        mask[i, :n] = True
        
    padded_inputs = {
        'crops': padded_crops,
        'pos': padded_pos,
        'labels': padded_labels,
        'mask': mask
    }
    
    return padded_inputs, padded_adj