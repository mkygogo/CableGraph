import cv2
from ultralytics import YOLO
from typing import List, Tuple
from PIL import Image
import os
import pathlib
import json
from ultralytics.engine.results import Results
from cable_inference import line_detect

def convert_to_gray(img_path):
    with Image.open(img_path) as img:
        # 转换为黑白图像
        black_and_white_img = img.convert('L')
        # 保存黑白图像
        black_and_white_img.save(img_path)    

def resize_img(img_path, target_size=(640, 640)):
    # 打开图像
    with Image.open(img_path) as img:
        # 获取图像的大小
        width, height = img.size
        print(f"原始图像大小: {width}x{height}")

        # 判断图像大小是否为640x640
        if (width, height) != target_size:
            print(f"图像大小不是{target_size}，正在转换...", )
            # 调整图像大小
            img = img.resize(target_size,  Image.Resampling.LANCZOS)
            
            # 构建新的文件名
            base, ext = os.path.splitext(img_path)
            new_filename = f"{base}_resized{ext}"
            
            # 保存调整后的图像
            img.save(new_filename)
            print(f"图像已转换并保存为: {new_filename}")
            return new_filename
        else:
            print(f"图像大小已经是{target_size}，无需转换。")
            return img_path  # 如果没有转换，返回原始文件名


def convert_yolo_obb_to_labelme(yolo_result: Results, 
                                  image_path_for_json: str,
                                  output_dir: str = ".",
                                  labelme_version: str = "2.3.6") -> str:
    """
    将单个YOLO OBB结果对象转换为LabelMe OBB JSON文件。
    
    此版本使用一个专用的 'image_path_for_json' 参数来
    1. 确定输出的 .json 文件名。
    2. 设置 JSON 文件内部的 'imagePath' 字段。
    这避免了依赖 'yolo_result.path'（该路径可能不正确）。

    Args:
        yolo_result: model.predict()返回的列表中的单个结果对象 (例如 results[0])。
        image_path_for_json: 您希望在JSON的 "imagePath" 字段中
                             写入的确切相对路径。
                             例如: "..\\images\\IMG_123.jpg"
                             函数将从此路径中提取文件名 "IMG_123.jpg"
                             来创建 "IMG_123.json"。
        output_dir: 用于保存生成的.json文件的目录。
        labelme_version: 要写入JSON文件中的LabelMe版本号。

    Returns:
        成功创建的JSON文件的完整路径。
    """
    
    # 1. 从结果对象中提取基本信息（不再使用 path）
    try:
        image_height, image_width = yolo_result.orig_shape[:2]
    except Exception as e:
        print(f"错误：无法访问 yolo_result 的属性 (orig_shape): {e}")
        print("请确保您传递的是一个有效的YOLO 'Results'对象 (例如 results[0])。")
        raise

    # 2. 准备输出文件路径 (使用新的 'image_path_for_json' 参数)
    #    例如: "IMG_123.jpg"
    image_filename = os.path.basename(image_path_for_json)
    
    #    例如: "IMG_123"
    base_filename = os.path.splitext(image_filename)[0]
    
    #    例如: "IMG_123.json"
    json_filename = f"{base_filename}.json"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    #    例如: "output_dir/IMG_123.json"
    output_path = os.path.join(output_dir, json_filename)


    # 3. 初始化 LabelMe JSON 结构
    labelme_data = {
        "version": labelme_version,
        "flags": {},
        "shapes": [],
        "imagePath": image_path_for_json, # 直接使用您提供的路径
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width)
    }

    # 4. 检查是否存在 OBB 数据
    if yolo_result.obb is None or len(yolo_result.obb) == 0:
        print(f"警告: {image_filename} 没有找到 OBB 结果。将创建一个空的JSON文件。")
    else:
        # 5. 获取类别名称和 OBB 数据
        class_names = yolo_result.names
        obb_data = yolo_result.obb

        box_points_list = obb_data.xyxyxyxy.cpu().numpy().tolist()
        angles = obb_data.xywhr[:, 4].cpu().numpy()
        class_indices = obb_data.cls.cpu().numpy().astype(int)

        # 6. 遍历每个检测到的目标并格式化
        for i in range(len(class_indices)):
            points = box_points_list[i]
            label = class_names[class_indices[i]]
            direction = angles[i]

            shape = {
                "label": str(label),
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "direction": float(direction)
            }
            
            labelme_data["shapes"].append(shape)

    # 7. 将数据写入JSON文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=4) 
    except Exception as e:
        print(f"错误：写入JSON文件到 {output_path} 时失败: {e}")
        raise

    # 8. 返回文件路径
    return output_path

yolo_model = YOLO('./weight/yolo_obb_best_0106.pt')
def yolo_img_detect(img) :
    img = resize_img(img)

    model = yolo_model
    
    im2 = cv2.imread(img)
    
    results = model.predict(source=im2, save=True, save_txt=True,iou=0.5) 

    if results :
        try:
            print(f"正在处理obb {img}...")
            json_relative_path = img.replace("./", "../", 1)
            json_file_path = convert_yolo_obb_to_labelme(
                results[0], 
                output_dir="output_jsons",
                image_path_for_json=json_relative_path # 匹配您示例中的 "..\images\..."
            )
            print(f"JSON 文件已成功保存到: {json_file_path}")
            return json_file_path, img
        except Exception as e:
            # 捕获 convert_yolo... 函数内部可能发生的任何其他错误
            print(f"为 {img} 生成JSON时出错: {e}")    

    else:
        # 'results' 列表是空的 (e.g., [])
        # 访问 results[0] 会导致 IndexError
        print(f"错误: model.predict() 未返回任何结果。跳过 {img}。")


def cable_detect(img_path):
    # 1. 执行 YOLO OBB 检测
    # yolo_img_detect 返回 (obb_json_path, resized_image_path)
    result = yolo_img_detect(img_path)
    if not result:
        return None
    
    obb_json_path, image_path = result

    # 2. 路径处理（适配 cable_inference）
    p = pathlib.Path(obb_json_path)
    posix_path = p.as_posix()
    usable_json_path = posix_path if posix_path.startswith(('./', '../', '/')) else f"./{posix_path}"

    # 3. 执行连线推理
    lines, final_image_path = line_detect(image_path, usable_json_path)

    # 4. 读取 OBB JSON 内容以返回给客户端
    with open(usable_json_path, 'r', encoding='utf-8') as f:
        obb_data = json.load(f)

    # 5. 组装最终结果
    final_result = {
        "obb_results": obb_data.get("shapes", []), # OBB 框及标签
        "line_results": lines,                     # 连线关系及置信度
        "image_path": final_image_path             # 最终绘制了连线的图片路径
    }
    
    return final_result

if __name__ == "__main__" :
    cable_detect("./files/IMG_20251022_114142.jpg")


