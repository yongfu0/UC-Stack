import cv2
import os

def extract_green_channel(image_path, output_path):
    # 读取TIFF图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 提取绿色通道
    green_channel = image.copy()
    green_channel[:, :, 0] = 0  # 将蓝色通道设为0
    green_channel[:, :, 2] = 0  # 将红色通道设为0
    # green_channel =  green_channel[:, :, 1]
    # 保存绿色通道图像
    cv2.imwrite(output_path, green_channel)

def process_images(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # 确保文件是TIFF图像文件
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 提取绿色通道并保存到另一文件夹
            extract_green_channel(image_path, output_path)



# 设置输入和输出文件夹路径
input_folder = "/home/ubuntu/fy/外部验证/eye_data/eye_ori_VB"
output_folder ="/home/ubuntu/fy/外部验证/eye_data/eye_seg_green"

# 执行提取绿色通道并保存操作
process_images(input_folder, output_folder)
