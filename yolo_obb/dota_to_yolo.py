from pathlib import Path
from ultralytics.utils import TQDM
import cv2
import os
def convert_dota_to_yolo_obb(dota_root_path: str,class_mapping:dict):
    
    """
    Converts DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Notes:
        The directory structure assumed for the DOTA dataset:

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)
    
    class_mapping = class_mapping

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        label_file = f"{image_name}.txt"  # 确保这行存在
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]

                # 添加调试信息 ， 打印出mapping文件里没有的错误标签                
                if class_name not in class_mapping:
                    print(f"错误: 在文件 {label_file} 发现未知类别 '{class_name}'")
                    #print(f"完整行内容: {line.strip()}")
                    continue
                # 调试结束
                
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")
    #转换label数据为yolo格式
    for phase in ["train", "val"]:
        
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase
        save_dir.mkdir(parents=True, exist_ok=True)
        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):

            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)

    #生成训练用的yaml文件
    yaml_path = "./YOLODataset/dataset.yaml"
    with open(yaml_path, 'w+') as yaml_file:
        yaml_file.write('train: %s\n' % \
                        os.path.abspath(os.path.join(dota_root_path, "images", "train")))
        yaml_file.write('val: %s\n\n' % \
                        os.path.abspath(os.path.join(dota_root_path, "images", "val")))
        yaml_file.write('nc: %i\n\n' % len(class_mapping.keys()))
        names_str = ''
        for label in class_mapping.keys():
            names_str += "'%s', " % label
        names_str = names_str.rstrip(', ')
        
        yaml_file.write('names: [%s]' % names_str)

#s设置自己标签的映射关系
#class_mapping = {
#        "L5_S1": 0,
#        "L4_L5": 1,
#    }

class_mapping =  {
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

#划分后的训练集和验证集的路径
#F:\MyProject\PythonProject\YOLOV8\data\SpineRDetection\YOLODataset
convert_dota_to_yolo_obb('./YOLODataset',class_mapping=class_mapping)

if __name__ == "__main__":
