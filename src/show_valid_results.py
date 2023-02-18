import glob
import os

import cv2
import matplotlib.pyplot as plt

RES_DIR = "andre50"


# Function to show validation predictions saved during training.
def show_valid_results(RES_DIR):
    EXP_PATH = f"data/yolo/images/val"
    validation_pred_images = glob.glob(f"{EXP_PATH}/*.jpg")
    print(validation_pred_images)
    for pred_image in validation_pred_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    show_valid_results(RES_DIR)
