import cv2
import numpy as np
import os

def convert_images(input_folder, output_folder):
    # 创建保存掩模图像的文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中的所有tif图像文件
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder,f))]

    for file in image_files:
        # 读取图像
        file_path = os.path.join(input_folder, file)
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)

         # 将图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 应用高斯滤波平滑图像
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

       # 应用自适应阈值处理图像
        _, thresholded = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

       # 执行形态学操作，填充孔洞
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 执行形态学操作，去除噪声
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # 执行图像边缘检测
        edges = cv2.Canny(opened, 0, 255)

        # 找到图像中的轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

       # 创建一个掩膜图像
        mask = np.zeros_like(image)

       # 绘制轮廓到掩膜图像上
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
       #
       #  #将掩膜应用到原始图像上
       #  masked_image = cv2.bitwise_and(image, mask)

        # 保存掩模图像到输出文件夹
        mask_file_path = os.path.join(output_folder, file)
        cv2.imwrite(mask_file_path, mask)
        print(f"处理完成，保存为 {mask_file_path}")



# 调用函数进行图像处理
# input_folder = r"E:\PycharmProjects\U_net_segmentation\SMDG\image"  # 替换为输入文件夹路径
# output_folder = r"E:\PycharmProjects\U_net_segmentation\SMDG\mask"  # 替换为输出文件夹路径
input_folder = "/home/ubuntu/fy/data/Messidor/image_224"
output_folder ="/home/ubuntu/fy/data/Messidor/mask"
convert_images(input_folder, output_folder)

