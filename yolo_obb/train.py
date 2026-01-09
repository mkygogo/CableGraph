from ultralytics import YOLO
import os

def train_obb():
    # 1. 初始化模型
    # 也可以选择 yolov8n-obb (更轻量) 或 yolov8x-obb (更精准)
    model = YOLO('yolov8m-obb.pt') 

    # 2. 开始训练
    results = model.train(
        # --- 核心路径 ---
        data="./YOLODataset/dataset.yaml",
        epochs=100,              # OBB任务较难收敛，建议 100 轮以上
        imgsz=640,               # 输入图像尺寸
        batch=16,                # 显存充足可尝试 32，显存不足设为 -1 自动匹配
        
        # --- 实验管理 ---
        project='Terminal_Project', # 项目文件夹名
        name='v8m_obb_base',        # 本次实验名称
        seed=2025,                  # 固定随机种子，方便复现实验结果
        deterministic=True,         # 保证结果可重复
        save=True,                  # 保存模型权重和训练日志
        save_period=10,             # 每 10 轮保存一个临时权重
        
        # --- 硬件优化 ---
        device=0,                   # 指定 GPU 编号
        workers=8,                  # Dataloader 线程数，通常设为 CPU 核心数
        amp=True,                   # 开启混合精度训练 (减少显存占用并加速)
        
        # --- 训练策略与超参数 ---
        optimizer='SGD',            # 默认 SGD，也可以选 'AdamW'
        lr0=0.01,                   # 初始学习率
        cos_lr=True,                # 使用余弦学习率调度
        patience=20,                # 早停机制：若 20 轮内 mAP 无提升则停止训练
        close_mosaic=10,            # 最后 10 轮关闭 Mosaic 增强，有助于模型权重收敛
        
        # --- 针对 OBB 的关键增强 (关键点) ---
        degrees=180.0,              # 允许 0-180 度全向旋转增强 (对旋转框至关重要)
        flipud=0.5,                 # 增加上下翻转概率
        fliplr=0.5,                 # 增加左右翻转概率
        mosaic=1.0,                 # 开启 Mosaic 增强，提升对小目标的检测能力
        mixup=0.1,                  # 混合增强，进一步防止过拟合
    )

    # 3. 验证与导出
    # 训练结束后导出为 TensorRT 或 ONNX 以备后续部署
    # model.export(format='engine', device=0)

if __name__ == '__main__':
    train_obb()
