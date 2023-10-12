import cv2
import numpy as np
import os

def remove_vessels(original_image, vessel_mask):
    # 将掩膜中的白色区域设为黑色
    result_image = original_image.copy()
    result_image[vessel_mask == 255] = [0, 0, 0]
    return result_image

def process_images(input_folder, mask_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # 确保文件是TIFF图像文件
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            mask_filename = 'result_processed_' + filename

            mask_path = os.path.join(mask_folder, mask_filename)
            print(mask_path)
            # 读取原始图像和掩膜图像
            original_image = cv2.imread(image_path)
            vessel_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 删除血管区域
            result_image = remove_vessels(original_image, vessel_mask)

            # 保存结果图像为TIFF格式
            cv2.imwrite(output_path, result_image)





# 设置输入和输出文件夹路径
input_folder= "/home/ubuntu/fy/外部验证/data/eye_tif"
mask_folder = "/home/ubuntu/fy/外部验证/data/eye_seg_VB"
output_folder = "/home/ubuntu/fy/外部验证/data/eye_seg_test"


# 执行血管删除操作
process_images(input_folder, mask_folder, output_folder)
