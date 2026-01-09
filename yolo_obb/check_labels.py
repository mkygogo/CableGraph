import os
import json

class_mapping = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9,
    "11": 10, "12": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18,
    "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "26": 25, "27": 26, "28": 27,
    "29": 28, "30": 29, "31": 30, "32": 31,
    "ua":32, "ub":33, "uc":34, "un":35,
    "ux":36, "uxn":37,
    "uu":38, "uv":39, "uw":40, "un3":41,
    "ia":42, "ia'":43, "ib":44, "ib'":45, "ic":46, "ic'":47,
    "3io":48, "io":49, "io'":50,
    "iu":51, "iu'":52, "iv":53, "iv'":54, "iw":55, "iw'":56,
    "kc01u":57, "kc01d":58, "kc02u":59, "kc02d":60, "kc03u":61, "kc03d":62, 
    "kc04u":63, "kc04d":64, "kc05u":65, "kc05d":66, "kc06u":67, "kc06d":68, 
    "cdyb":69, "com1":70, "com2":71, 
    "kc09":72, "kc10":73, "kc11":74, "kc12":75, "kc13":76,
    "krau":77, "krad":78, "krbu":79, "krbd":80, "krcu":81, "krcd":82,
    "zd1":83, "zd2":84, "kd1":85, "kd2":86,
    "dg":87, "yf": 88
}

def check_json_labels(directory_path):
    """检查目录下所有JSON文件中的label是否在class_mapping中"""
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    print(f"在目录 '{directory_path}' 中找到 {len(json_files)} 个JSON文件")
    print("正在检查...\n")
    
    problematic_files = []
    all_found_labels = set()
    unknown_labels = set()
    
    for json_file in json_files:
        json_path = os.path.join(directory_path, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查每个shape的label
            has_unknown_label = False
            for shape in data.get('shapes', []):
                label = shape.get('label', '')
                all_found_labels.add(label)
                
                if label not in class_mapping:
                    unknown_labels.add(label)
                    has_unknown_label = True
            
            # 如果有未知label，记录文件信息
            if has_unknown_label:
                problematic_files.append({
                    'file': json_file,
                    'imagePath': data.get('imagePath', ''),
                    'unknown_labels': [label for shape in data.get('shapes', []) 
                                     if shape.get('label', '') not in class_mapping]
                })
                
        except Exception as e:
            print(f"读取文件 {json_file} 时出错: {e}")
    
    # 输出结果
    print("=" * 50)
    print("检查结果:")
    print("=" * 50)
    
    print(f"\n所有发现的label: {sorted(all_found_labels)}")
    print(f"未知的label: {sorted(unknown_labels)}")
    
    if problematic_files:
        print(f"\n发现 {len(problematic_files)} 个文件包含未知label:")
        print("-" * 50)
        
        for file_info in problematic_files:
            print(f"文件: {file_info['file']}")
            print(f"图片路径: {file_info['imagePath']}")
            print(f"未知label: {file_info['unknown_labels']}")
            print("-" * 30)
    #else:
    #    print("\n✓ 所有文件的label都在class_mapping中！")
    
    return problematic_files, all_found_labels, unknown_labels

# 使用示例
if __name__ == "__main__":
    directory_path = "./label"  # 替换为你的目录路径
    problematic_files, all_found_labels, unknown_labels = check_json_labels(directory_path)