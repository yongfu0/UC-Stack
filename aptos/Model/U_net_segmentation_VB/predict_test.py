import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from Src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model.pth"
    img_folder = "/home/ubuntu/fy/外部验证/eye_data/pro_eye_224"
    roi_mask_folder = "/home/ubuntu/fy/外部验证/eye_data/eye_mask_224"

    save_folder = "/home/ubuntu/fy/外部验证/eye_data/eye_VB"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_folder), f"image folder {img_folder} not found."
    assert os.path.exists(roi_mask_folder), f"mask folder {roi_mask_folder} not found."

    # mean = (0.709, 0.381, 0.224)
    # std = (0.127, 0.079, 0.043)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # Create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # Get image and mask file paths
    img_paths = sorted([os.path.join(img_folder, img_name) for img_name in os.listdir(img_folder)])
    mask_paths = sorted([os.path.join(roi_mask_folder, mask_name) for mask_name in os.listdir(roi_mask_folder)])

    for img_path, mask_path in zip(img_paths, mask_paths):
        # Load ROI mask
        roi_img = Image.open(mask_path).convert('L')
        roi_img = np.array(roi_img)

        # Load image
        original_img = Image.open(img_path).convert('RGB')

        # From PIL image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # Expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # Enter evaluation mode
        with torch.no_grad():
            # Initialize model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("Inference time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            # Set pixel values corresponding to foreground to 255 (white)
            prediction[prediction == 1] = 255
            # Set pixel values in uninterested regions to 0 (black)
            prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)

            # Save result
            # save_path = os.path.join(img_folder, "result_" + os.path.basename(img_path))
            save_path = os.path.join(save_folder, "result_" + os.path.basename(img_path))
            mask.save(save_path)
            print("Saved result at", save_path)


if __name__ == '__main__':
    main()
