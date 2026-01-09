# cable_inference.py
import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from scipy.spatial.distance import cdist

from cable_model import CableLinkModel
from cable_data_loader import _get_obb_crop # 复用数据加载器中的函数

def load_model_and_vocab(model_path, vocab_path, device):
    """加载模型和词汇表"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # 假设模型参数与训练时一致
    model = CableLinkModel(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, vocab

def predict_links(image_path, label_json_path, model, vocab, device):
    """
    对单张图片进行推理，并绘制结果。
    (已修复数据污染BUG)
    """
    
    # 1. 定义图像转换
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. 加载图像和标签
    # *** 关键修复：创建两个图像副本 ***
    # image_clean 用于提取特征 (永远不修改)
    image_clean = Image.open(image_path).convert("RGB")
    # image_to_draw 用于绘制 (将被修改)
    image_to_draw = image_clean.copy() 
    
    draw = ImageDraw.Draw(image_to_draw)
    img_w, img_h = image_clean.size
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()

    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    nodes = []
    node_labels_str = [] 
    node_centers = [] 
    
    # 3. 提取节点特征并绘制OBB和标签
    print("Extracting features and drawing OBBs/labels...")
    for shape in label_data.get('shapes', []):
        points = shape['points']
        center = np.mean(points, axis=0)
        
        # --- 绘制 (在 image_to_draw 上) ---
        draw.polygon([tuple(p) for p in points], outline="red", width=2)
        label_str = shape['label']
        draw.text((center[0] + 5, center[1] - 10), label_str, fill="red", font=font)

        # --- 准备模型输入 (从 image_clean 提取) ---
        # *** 关键修复：使用干净的图像 'image_clean' ***
        crop = _get_obb_crop(image_clean, points) 
        crop_tensor = img_transform(crop)
        
        pos_tensor = torch.tensor([center[0] / img_w, center[1] / img_h], dtype=torch.float)
        
        label_index = vocab.get(label_str, vocab['<UNK>'])
        label_tensor = torch.tensor(label_index, dtype=torch.long)
        
        nodes.append({
            'crop': crop_tensor,
            'pos': pos_tensor,
            'label': label_tensor
        })
        node_labels_str.append(label_str)
        node_centers.append(tuple(center)) 

    if not nodes:
        print("No OBBs found in label file.")
        return [], None

    # 4. 手动组合成一个批次 (B=1)
    num_nodes = len(nodes)
    crops = torch.stack([n['crop'] for n in nodes]).unsqueeze(0) 
    pos = torch.stack([n['pos'] for n in nodes]).unsqueeze(0) 
    labels = torch.stack([n['label'] for n in nodes]).unsqueeze(0) 
    mask = torch.ones_like(labels, dtype=torch.bool)
    
    inputs = {
        'crops': crops.to(device),
        'pos': pos.to(device),
        'labels': labels.to(device),
        'mask': mask.to(device)
    }

    # 5. 模型推理
    print("Running model inference...")
    with torch.no_grad():
        logits = model(**inputs) 
    
    # 6. 后处理
    probs = torch.sigmoid(logits.squeeze(0)) 
    preds = (probs > 0.5).int().cpu().numpy()

    # 7. 格式化输出并绘制预测的连线
    print("Drawing predicted connections...")
    connections = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): 
            if preds[i, j] == 1:
                connections.append({
                    'from': node_labels_str[i],
                    'to': node_labels_str[j],
                    'confidence': probs[i, j].item()
                })
                
                # 绘制连线 (在 image_to_draw 上)
                center_i = node_centers[i]
                center_j = node_centers[j]
                draw.line([center_i, center_j], fill="green", width=3)
                
    # 8. 保存图像
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    output_filename = f"{base_name}_output{ext}"
    output_path = os.path.join(".", output_filename) 
    
    # *** 关键修复：保存被绘制的副本 'image_to_draw' ***
    image_to_draw.save(output_path)
    print(f"Output image saved to: {output_path}")
                
    return connections, output_path

if __name__ == "__main__":
    # --- 配置推理参数 ---
    MODEL_PATH = "checkpoints/best_model.pth"
    VOCAB_PATH = "checkpoints/vocab.json"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 指定要推理的图像 ---
    FILE = "IMG_20251106_103114"
    TEST_IMAGE_PATH = f"test/images/{FILE}.jpg"
    TEST_LABEL_PATH = f"test/label/{FILE}.json"

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print(f"Error: Model ('{MODEL_PATH}') or vocab ('{VOCAB_PATH}') not found.")
        print("Please run cable_train.py first.")
    else:
        print("Loading model and vocab...")
        model, vocab = load_model_and_vocab(MODEL_PATH, VOCAB_PATH, DEVICE)
        
        print(f"Running inference on {TEST_IMAGE_PATH}...")
        predicted_connections, saved_image_path = predict_links(
            TEST_IMAGE_PATH, TEST_LABEL_PATH, model, vocab, DEVICE
        )
        
        print("\n--- Predicted Connections ---")
        if predicted_connections:
            for conn in predicted_connections:
                print(f"  {conn['from']} <--> {conn['to']} (Confidence: {conn['confidence']:.4f})")
        else:
            print("  No connections predicted.")
        
        if saved_image_path:
            print(f"\nVisual output saved to: {saved_image_path}")