import os
import random
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
# from IPython.display import Image  # for displaying images
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from tqdm import tqdm

random.seed(42)

config_path = "data/yolo/config.yaml"
with open(config_path) as file:
    config = yaml.load(file, yaml.CLoader)

class_id_to_name_mapping = {i: class_name for i, class_name in enumerate(config['names'])}
class_name_to_id_mapping = {class_name: i for i, class_name in class_id_to_name_mapping.items()}

COLORS = ("orange", "green", "blue", "purple", "yellow",)

print(class_id_to_name_mapping)
print(class_name_to_id_mapping)


def plot_bounding_box(image, annotation_list, width=3):
    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (transformed_annotations[:, 3] / 2)
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (transformed_annotations[:, 4] / 2)
    transformed_annotations[:, 3] = transformed_annotations[:, 1] + transformed_annotations[:, 3]
    transformed_annotations[:, 4] = transformed_annotations[:, 2] + transformed_annotations[:, 4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann

        obj_cls = int(obj_cls)
        x0, y0 = x0 - width, y0 - width
        x1, y1 = x1 + width, y1 + width

        plotted_image.rectangle(((x0, y0), (x1, y1)), outline=COLORS[obj_cls], width=width)
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[obj_cls], fill=COLORS[obj_cls])

    # image.format = 'png'
    plt.imshow(np.array(image))
    # plt.imsave("supsup.jpg", np.array(image))

    plt.show()


# Get any random annotation file
subdir, filename = "val", "page-1"
subdir, filename = "train", "page-3"
IMAGE_FILE = os.path.join(config[subdir], f"{filename}.jpg")
LABEL_FILE = IMAGE_FILE.replace("images", "labels").replace("jpg", "txt")

with open(LABEL_FILE, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split("\t") for x in annotation_list]
    annotation_list = [[float(y) for y in x] for x in annotation_list]

# Get the corresponding image file
assert os.path.exists(IMAGE_FILE)

# Load the image
image = Image.open(IMAGE_FILE)

# Plot the Bounding Box
plot_bounding_box(image, annotation_list)
