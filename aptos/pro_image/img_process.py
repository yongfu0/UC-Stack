import os
import shutil
import pandas as pd
from PIL import Image

def split_images_by_label(csv_file, image_dir, output_dir, target_size):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 遍历CSV文件中的每一行
    for index, row in df.iterrows():
        # 获取图像名和标签
        # image_name = row['Image name']
        # label = row['Retinopathy grade']
        image_name = row['image']
        label = row['level']

        # 创建标签文件夹（如果不存在）
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # 加载图像并调整大小
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        image = image.resize(target_size)

        # # 生成新的图像名，并保存到标签文件夹
        # new_image_name = image_name.split('.')[0] + '.jpg'
        # new_image_path = os.path.join(label_dir, new_image_name)
        # image.save(new_image_path, 'JPEG')
        new_image_path = os.path.join(label_dir, image_name)
        image.save(new_image_path)



# 示例用法

csv_file = "/home/ubuntu/fy/外部验证/eyepacs_preprocess/trainLabels.csv"
image_dir = "/home/ubuntu/fy/外部验证/data/eye_seg_green"
output_dir = "/home/ubuntu/fy/外部验证/data/eye_seg_gro"
target_size = (224, 224)

# 将不同标签的图片分别放入不同文 件夹中并调整大小
split_images_by_label(csv_file, image_dir, output_dir, target_size)
