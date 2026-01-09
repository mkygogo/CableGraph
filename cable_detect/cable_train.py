# cable_train.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score

from cable_data_loader import build_vocab, CableDataset, collate_fn
from cable_model import CableLinkModel

# --- 配置 ---
class HP:
    # 路径
    ROOT_DIR = "./" # 假设的根目录
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    LABEL_DIR = os.path.join(ROOT_DIR, "label")
    LINE_DIR = os.path.join(ROOT_DIR, "label_line")
    SPLIT_DIR = os.path.join(ROOT_DIR, "dataset_splits")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    
    # 超参数
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 200
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 不平衡比例 (假设非连接是连接的20倍, 这是一个估计值, 最好在数据集上计算)
    POS_WEIGHT = 20.0 
    
    # 模型参数
    EMBED_DIM = 64
    VISUAL_DIM = 128
    TRANSFORMER_DIM = 256

# --- 辅助函数 ---
def load_split_files(split_dir):
    # 假设分割文件为 train.txt 和 val.txt
    def read_file(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
            
    train_files = read_file(os.path.join(split_dir, "train.txt"))
    val_files = read_file(os.path.join(split_dir, "test.txt"))
    return train_files, val_files

def calculate_metrics(preds, targets):
    preds = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    return f1, precision, recall

# --- 主训练函数 ---
def main():
    print(f"Using device: {HP.DEVICE}")
    os.makedirs(HP.OUTPUT_DIR, exist_ok=True)
    
    # 1. 构建词汇表
    print("Building vocab...")
    vocab = build_vocab(HP.LABEL_DIR)
    vocab_path = os.path.join(HP.OUTPUT_DIR, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocab size: {len(vocab)}. Saved to {vocab_path}")
    
    # 2. 加载数据
    train_files, val_files = load_split_files(HP.SPLIT_DIR)
    print(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CableDataset(HP.IMAGE_DIR, HP.LABEL_DIR, HP.LINE_DIR, train_files, vocab, img_transform)
    val_dataset = CableDataset(HP.IMAGE_DIR, HP.LABEL_DIR, HP.LINE_DIR, val_files, vocab, img_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=HP.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=HP.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 3. 初始化模型、损失函数、优化器
    model = CableLinkModel(
        vocab_size=len(vocab),
        embed_dim=HP.EMBED_DIM,
        visual_dim=HP.VISUAL_DIM,
        transformer_dim=HP.TRANSFORMER_DIM
    ).to(HP.DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([HP.POS_WEIGHT]).to(HP.DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=HP.LEARNING_RATE)
    
    # 4. 训练循环
    best_val_f1 = 0.0
    
    for epoch in range(HP.NUM_EPOCHS):
        # --- 训练 ---
        model.train()
        total_train_loss = 0
        
        for inputs, targets in train_loader:
            if inputs['crops'].numel() == 0: continue # 跳过空批次
                
            inputs = {k: v.to(HP.DEVICE) for k, v in inputs.items()}
            targets = targets.to(HP.DEVICE)
            
            optimizer.zero_grad()
            logits = model(**inputs)
            
            # 只在有效掩码区域计算损失
            pair_mask = inputs['mask'].unsqueeze(2) * inputs['mask'].unsqueeze(1)
            valid_logits = logits[pair_mask]
            valid_targets = targets[pair_mask]
            
            loss = criterion(valid_logits, valid_targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证 ---
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                if inputs['crops'].numel() == 0: continue
                
                inputs = {k: v.to(HP.DEVICE) for k, v in inputs.items()}
                targets = targets.to(HP.DEVICE)
                
                logits = model(**inputs)
                
                pair_mask = inputs['mask'].unsqueeze(2) * inputs['mask'].unsqueeze(1)
                valid_logits = logits[pair_mask]
                valid_targets = targets[pair_mask]
                
                loss = criterion(valid_logits, valid_targets)
                total_val_loss += loss.item()
                
                # 计算指标
                preds = (torch.sigmoid(valid_logits) > 0.5).int()
                all_preds.append(preds)
                all_targets.append(valid_targets)
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        if all_preds:
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            val_f1, val_p, val_r = calculate_metrics(all_preds, all_targets)
        else:
            val_f1, val_p, val_r = 0, 0, 0
            
        print(f"Epoch {epoch+1}/{HP.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Val P: {val_p:.4f} | Val R: {val_r:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(HP.OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (F1: {best_val_f1:.4f})")

if __name__ == "__main__":
    main()