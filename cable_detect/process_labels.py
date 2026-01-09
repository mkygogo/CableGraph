import os
import json

# --- 配置 ---
INPUT_DIR = 'temp/label'
OUTPUT_DIR = 'temp/label_line'
# --- 结束配置 ---

def filter_l_labels():
    """
    遍历 INPUT_DIR 中的所有 JSON 文件，
    1. 将 'shapes' 列表中 'label' 为 'l' 的项保存到 OUTPUT_DIR 中的同名文件
    2. 从原始文件中移除 'label' 为 'l' 的项
    """
    
    # 1. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"开始处理目录 '{INPUT_DIR}'...")
    
    # 2. 遍历输入目录中的所有文件
    try:
        filenames = os.listdir(INPUT_DIR)
    except FileNotFoundError:
        print(f"错误：输入目录 '{INPUT_DIR}' 不存在。")
        return
    except NotADirectoryError:
        print(f"错误：'{INPUT_DIR}' 不是一个目录。")
        return

    processed_count = 0
    skipped_count = 0

    for filename in filenames:
        # 确保只处理 JSON 文件
        if not filename.endswith('.json'):
            continue
            
        input_filepath = os.path.join(INPUT_DIR, filename)
        output_filepath = os.path.join(OUTPUT_DIR, filename)
        
        try:
            # 3. 读取原始 JSON 文件
            with open(input_filepath, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)
            
            # 4. 复制原始数据结构以保留所有其他键
            output_data = data.copy()
            original_data = data.copy()  # 用于修改原始文件
            
            # 5. 处理 'shapes' 列表
            if 'shapes' in data and isinstance(data['shapes'], list):
                # 筛选出所有 label 为 'l' 的 shape（用于输出到label_line）
                l_shapes = [
                    shape for shape in data['shapes'] 
                    if shape.get('label') == 'l'
                ]
                
                # 移除所有 label 为 'l' 的 shape（用于修改原始文件）
                filtered_shapes = [
                    shape for shape in data['shapes'] 
                    if shape.get('label') != 'l'
                ]
                
                # 6. 更新数据
                # 6.1 将"l"标签保存到label_line目录
                output_data['shapes'] = l_shapes
                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    json.dump(output_data, f_out, indent=2)
                
                # 6.2 从原始文件中移除"l"标签
                original_data['shapes'] = filtered_shapes
                with open(input_filepath, 'w', encoding='utf-8') as f_original:
                    json.dump(original_data, f_original, indent=2)
                
                print(f"  [成功] 已处理 '{filename}'")
                processed_count += 1
                
            else:
                print(f"  [跳过] '{filename}' 中没有 'shapes' 列表。")
                skipped_count += 1

        except json.JSONDecodeError:
            print(f"  [错误] '{filename}' 不是有效的 JSON 文件。")
            skipped_count += 1
        except Exception as e:
            print(f"  [错误] 处理 '{filename}' 时发生未知错误: {e}")
            skipped_count += 1
            
    print("\n--- 处理完成 ---")
    print(f"总共成功处理: {processed_count} 个文件")
    print(f"总共跳过/失败: {skipped_count} 个文件")
    print(f"'l' 标签已保存至 '{OUTPUT_DIR}' 目录")
    print(f"原始文件中的 'l' 标签已移除")

# --- 运行脚本 ---
if __name__ == "__main__":
    filter_l_labels()