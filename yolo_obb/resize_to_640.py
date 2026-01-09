from PIL import Image
import os

def resize_imgs():
    # 设置目标大小
    target_size = (640, 640)

    # 获取当前目录下的所有 .jpg 文件
    for filename in os.listdir('./images'):
        if filename.endswith('.jpg'):
            with Image.open(filename) as img:
                # 调整图像大小
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                # 保存调整后的图像，覆盖原始文件
                # 如果你想保留原始文件，可以保存到不同的文件名或目录
                resized_img.save(filename)

    print("所有 .jpg 文件已调整为 640x640 大小。")   



def resize_all_imgs(input_dir='./images', output_dir_640='./640images',  
                    output_dir_1024='./1024images', output_dir_2048='./2048images',
                    output_dir_3072='./3072images'):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir_640, exist_ok=True)
    os.makedirs(output_dir_1024, exist_ok=True)
    os.makedirs(output_dir_2048, exist_ok=True)
    os.makedirs(output_dir_3072, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 构建完整的文件路径
        input_path = os.path.join(input_dir, filename)
        
        # 检查文件是否为图像文件（可以根据需要添加更多格式）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            output_path_640 = os.path.join(output_dir_640, filename)
            output_path_1024 = os.path.join(output_dir_1024, filename)
            output_path_2048 = os.path.join(output_dir_2048, filename)
            output_path_3072 = os.path.join(output_dir_3072, filename)

            resize_image(input_path, output_path_640,  
                         output_path_1024, output_path_2048, output_path_3072)


def resize_image(image_path, output_path_640, output_path_1024, output_path_2048, output_path_3072):
    target_size_640 = (640, 640)
    target_size_1024 = (1024, 1024)
    target_size_2048 = (2048, 2048)
    target_size_3072 = (3072, 3072)
    # 打开图像
    with Image.open(image_path) as img:
        # 获取图像的大小
        width, height = img.size
        print(f"原始图像大小: {width}x{height}")

        # 判断图像大小是否为640x640
        if (width, height) != target_size_640:
            print("图像大小不是640x640，正在转换...")
            # 调整图像大小
            img = img.resize(target_size_640, Image.Resampling.LANCZOS)
            # 保存调整后的图像
            img.save(output_path_640)
            print(f"图像已转换并保存为: {output_path_640}")
        else:
            img.save(output_path_640)
            print("图像大小已经是640x640，无需转换。")

        # 判断图像大小是否为1024x1024
        if (width, height) != target_size_1024:
            print("图像大小不是1024x1024，正在转换...")
            # 调整图像大小
            img = img.resize(target_size_1024, Image.Resampling.LANCZOS)
            # 保存调整后的图像
            img.save(output_path_1024)
            print(f"图像已转换并保存为: {output_path_1024}")
        else:
            img.save(output_path_1024)
            print("图像大小已经是1024x1024，无需转换。")

        # 判断图像大小是否为2048
        if (width, height) != target_size_2048:
            print("图像大小不是2048x2048，正在转换...")
            # 调整图像大小
            img = img.resize(target_size_2048, Image.Resampling.LANCZOS)
            # 保存调整后的图像
            img.save(output_path_2048)
            print(f"图像已转换并保存为: {output_path_2048}")
        else:
            img.save(output_path_2048)
            print("图像大小已经是2048x2048，无需转换。")

        # 判断图像大小是否为3072
        if (width, height) != target_size_3072:
            print("图像大小不是3072x3072，正在转换...")
            # 调整图像大小
            img = img.resize(target_size_3072, Image.Resampling.LANCZOS)
            # 保存调整后的图像
            img.save(output_path_3072)
            print(f"图像已转换并保存为: {output_path_3072}")
        else:
            img.save(output_path_3072)
            print("图像大小已经是3072x3072，无需转换。")            




def resize_img(img_path, target_size=(640, 640)):
    # 打开图像
    with Image.open(img_path) as img:
        # 获取图像的大小
        width, height = img.size
        print(f"原始图像大小: {width}x{height}")

        # 判断图像大小是否为640x640
        if (width, height) != target_size:
            print("图像大小不是640x640，正在转换...")
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
            print("图像大小已经是640x640，无需转换。")
            return img_path  # 如果没有转换，返回原始文件名

if __name__ == "__main__":
    print("将images目录下的所有图片转换成640*640")
    resize_all_imgs()
    #input_image_path = "images/test1.jpg"  # 替换为你的图片路径
    #resized_image_path = resize_img(input_image_path)
    #print("转换后的文件："+resized_image_path)