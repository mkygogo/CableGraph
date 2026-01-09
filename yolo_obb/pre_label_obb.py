import cv2
from ultralytics import YOLO
from typing import List, Tuple
from PIL import Image
import os
import pathlib
import json
from ultralytics.engine.results import Results


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

yolo_model = YOLO('./yolo_obb_best_1112.pt')
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
                output_dir="label",
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

#from cable_inference import line_detect

# if __name__ == "__main__" :
#     obb_json_path, image_path = yolo_img_detect("./images/IMG_20251106_092359.jpg")
#     print(obb_json_path)
#     print(image_path)

#     windows_path_str = obb_json_path
#     p = pathlib.Path(windows_path_str)
#     #  使用 .as_posix() 方法将其转换为使用正斜杠 (/) 的字符串
#     posix_path = p.as_posix()
#     # 手动添加 "./" 前缀
#     if not posix_path.startswith(('./', '../', '/')):
#         usable_path = f"./{posix_path}"
#     else:
#         usable_path = posix_path

#     print(f"原始路径: {windows_path_str}")
#     print(f"可用路径: {usable_path}")

#     #line_detect(image_path, usable_path)


import pathlib

def process_image(image_file_path):
    """
    处理单张图片的函数，包含您原来的逻辑。
    """
    try:
        # 1. 调用 YOLO 检测
        # 注意：我们将 Path 对象转换为字符串，以匹配您原始代码的行为
        obb_json_path, image_path = yolo_img_detect(str(image_file_path))
        
        print(f"  - JSON Path: {obb_json_path}")
        print(f"  - Image Path: {image_path}")

        # 2. 转换 JSON 路径
        windows_path_str = obb_json_path
        p = pathlib.Path(windows_path_str)
        
        # 使用 .as_posix() 方法将其转换为使用正斜杠 (/) 的字符串
        posix_path = p.as_posix()
        
        # 手动添加 "./" 前缀
        if not posix_path.startswith(('./', '../', '/')) and posix_path:
            usable_path = f"./{posix_path}"
        else:
            usable_path = posix_path

        print(f"  - 原始路径: {windows_path_str}")
        print(f"  - 可用路径: {usable_path}")

    except Exception as e:
        print(f"  - 处理文件 {image_file_path} 时出错: {e}")

if __name__ == "__main__":
    # 1. 定义要遍历的目录
    image_directory = pathlib.Path("./images")
    
    # 2. 定义支持的图片扩展名
    # 您可以根据需要添加或删除扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    print(f"--- 开始处理目录: {image_directory} ---")

    # 3. 检查目录是否存在
    if not image_directory.is_dir():
        print(f"错误: 目录 '{image_directory}' 不存在或不是一个目录。")
    else:
        # 4. 遍历目录中的所有文件
        file_count = 0
        for file_path in image_directory.glob('*'):
            
            # 5. 检查是否是文件，并且文件扩展名在我们的支持列表中
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                file_count += 1
                print(f"\n[处理第 {file_count} 张图片: {file_path.name}]")
                
                # 6. 调用处理函数
                process_image(file_path)

        if file_count == 0:
            print("在目录中未找到支持的图片文件。")
        else:
            print(f"\n--- 处理完毕，共处理 {file_count} 张图片 ---")
