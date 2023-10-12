import os
import cv2
import numpy as np

def apply_clahe(img):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return img_clahe

def apply_gaussian_blur(img):
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    return img_blur

def homofilter(I):
    I = np.double(I)
    m, n = I.shape
    rL = 0.5
    rH = 2
    c = 2
    d0 = 20
    I1 = np.log(I + 1)
    FI = np.fft.fft2(I1)
    n1 = np.floor(m / 2)
    n2 = np.floor(n / 2)
    D = np.zeros((m, n))
    H = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = ((i - n1) ** 2 + (j - n2) ** 2)
            H[i, j] = (rH - rL) * (np.exp(c * (-D[i, j] / (d0 ** 2)))) + rL
    I2 = np.fft.ifft2(H * FI)
    I3 = np.real(np.exp(I2) - 1)
    I4 = I3 - np.min(I3)
    I4 = I4 / np.max(I4) * 255
    dstImg = np.uint8(I4)
    return dstImg



def process_image(img_path):
    # Read input image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE histogram equalization
    img_clahe =apply_clahe(img)

    # Apply Gaussian filter
    img_gaussian = cv2.GaussianBlur(img_clahe, (5, 5), 0)

    # Apply homomorphic filtering
    img_filtered = homofilter(img_gaussian)


    # Convert to RGB for visualization
    img_rgb = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2RGB)

    return img_rgb

if __name__ == '__main__':
    # Set the input folder path
    folder_path = "/home/ubuntu/fy/data/Messidor/image"

    # Create an output folder for the results

    processed_output_folder = "/home/ubuntu/fy/data/Messidor/processed/"
    os.makedirs(processed_output_folder, exist_ok=True)

    # Get a list of image files in the input folder
    image_files = os.listdir(folder_path)

    for file_name in image_files:
        # if file_name.endswith('.tif') or file_name.endswith('.png'):
        # Construct the full path of the image file
        img_path = os.path.join(folder_path, file_name)

        # Process the image
        img_rgb = process_image(img_path)


        # Save the mask and processed RGB image

        rgb_path = os.path.join(processed_output_folder, f'processed_{file_name}')

        cv2.imwrite(rgb_path, img_rgb)

        print(f"Processed image: {file_name}")

    print("Processing complete.")
