from tqdm import tqdm
import shutil
import os
from sklearn.model_selection import train_test_split

def CollateDataset(image_dir,label_dir,val_size = 0.2,random_state=42):  # image_dir:图片路径  label_dir：标签路径
    if not os.path.exists("./YOLODataset"):
        os.makedirs("./YOLODataset")

    images = []
    labels = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        ext = os.path.splitext(image_name)[-1]
        label_name = image_name.replace(ext, ".txt")
        label_path = os.path.join(label_dir, label_name)
        if not os.path.exists(label_path):
            print("there is no:", label_path)
        else:
            images.append(image_path)
            labels.append(label_path)
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=val_size, random_state=random_state)


    destination_images = "./YOLODataset/images"
    destination_labels = "./YOLODataset/labels"
    os.makedirs(os.path.join(destination_images, "train"), exist_ok=True)
    os.makedirs(os.path.join(destination_images, "val"), exist_ok=True)
    os.makedirs(os.path.join(destination_labels, "train_original"), exist_ok=True)
    os.makedirs(os.path.join(destination_labels, "val_original"), exist_ok=True)
    # 遍历每个有效图片路径
    for i in tqdm(range(len(train_data))):
        image_path = train_data[i]
        label_path = train_labels[i]

        image_destination_path = os.path.join(destination_images, "train", os.path.basename(image_path))
        shutil.copy(image_path, image_destination_path)
        label_destination_path = os.path.join(destination_labels, "train_original", os.path.basename(label_path))
        shutil.copy(label_path, label_destination_path)

    for i in tqdm(range(len(test_data))):
        image_path = test_data[i]
        label_path = test_labels[i]
        image_destination_path = os.path.join(destination_images, "val", os.path.basename(image_path))
        shutil.copy(image_path, image_destination_path)
        label_destination_path = os.path.join(destination_labels, "val_original", os.path.basename(label_path))
        shutil.copy(label_path, label_destination_path)

if __name__ == '__main__':
    CollateDataset("./imgs","./labelTxt")