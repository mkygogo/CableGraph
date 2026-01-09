# process_dataset.py

import os
import json
import glob
import random
from typing import List

# --- 配置 ---
# 请确保这些目录路径与您的项目结构一致
LABEL_DIR = './label'
LABEL_LINE_DIR = './label_line'
TRAIN_RATIO = 0.8 # 80% 用于训练
# 分割列表的输出目录
OUTPUT_DIR = './dataset_splits'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.txt')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test.txt')

def get_valid_base_names(label_dir: str, label_line_dir: str) -> List[str]:
    """
    遍历 OBB 标签目录，找到有对应线条 JSON 文件的 OBB 文件的基础文件名。
    返回的文件名不包含扩展名，如 'IMG_123'。
    """
    base_names = []
    
    # 查找所有 OBB JSON 文件
    obb_json_files = glob.glob(os.path.join(label_dir, '*.json'))
    
    for obb_file_path in obb_json_files:
        obb_base_name_with_ext = os.path.basename(obb_file_path)
        
        # 移除 .json 得到基础文件名，例如 'IMG_123'
        obb_base_name = obb_base_name_with_ext.replace('.json', '')
        
        # 构造对应的线条 JSON 文件路径
        # 假设：IMG_123.json 对应 IMG_123_line.json
        line_file_name = obb_base_name_with_ext#.replace('.json', '_line.json')
        line_file_path = os.path.join(label_line_dir, line_file_name)
        
        # 检查线条文件是否存在
        if os.path.exists(line_file_path):
            base_names.append(obb_base_name)
        # else:
        #     # 可以在这里打印警告，提示缺少线条文件
        #     pass

    return base_names

def split_and_save_dataset(base_names: List[str], train_ratio: float, output_dir: str, train_file: str, test_file: str):
    """
    将文件名列表分割为训练集和测试集，并保存到文件中。
    """
    
    # 设置随机种子以保证分割结果可复现
    random.seed(42) 
    random.shuffle(base_names)
    
    num_total = len(base_names)
    num_train = int(num_total * train_ratio)
    
    train_set = base_names[:num_train]
    test_set = base_names[num_train:]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集列表
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_set))
        
    # 保存测试集列表
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_set))
        
    print("--- 数据集分割完成 ---")
    print(f"总文件数: {num_total}")
    print(f"训练集文件数 ({train_ratio*100:.0f}%): {len(train_set)}")
    print(f"测试集文件数 ({(1-train_ratio)*100:.0f}%): {len(test_set)}")
    print(f"训练列表已保存到: {train_file}")
    print(f"测试列表已保存到: {test_file}")


def main():
    print("--- 正在获取有效的基础文件名 ---")
    base_names = get_valid_base_names(LABEL_DIR, LABEL_LINE_DIR)
    
    if not base_names:
        print("错误：未找到任何有效的 OBB/线条 JSON 文件对。请检查目录路径和文件命名规则。")
        return
        
    print(f"找到 {len(base_names)} 个匹配的文件对。")
    
    # 执行分割和保存
    split_and_save_dataset(base_names, TRAIN_RATIO, OUTPUT_DIR, TRAIN_FILE, TEST_FILE)


if __name__ == '__main__':
    main()