import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw


def plot_bounding_box(image, annotation_list, width=3, colors=None, save_only=False):
    annotations = np.array(annotation_list)
    w, h = image.size
    print(w, h)

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    # transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    # transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (
        transformed_annotations[:, 3] / 2
    )
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (
        transformed_annotations[:, 4] / 2
    )
    transformed_annotations[:, 3] = (
        transformed_annotations[:, 1] + transformed_annotations[:, 3]
    )
    transformed_annotations[:, 4] = (
        transformed_annotations[:, 2] + transformed_annotations[:, 4]
    )

    if colors is None:
        colors = COLORS

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann

        obj_cls = int(obj_cls)
        x0, y0 = x0 - width, y0 - width
        x1, y1 = x1 + width, y1 + width

        plotted_image.rectangle(
            ((x0, y0), (x1, y1)), outline=colors[obj_cls], width=width
        )
        # plotted_image.text(
        #     (x0, y0 - 10), class_id_to_name_mapping[obj_cls], fill=colors[obj_cls]
        # )

    # image.format = 'png'
    if save_only:
        plt.imsave("out/saved.jpg", np.array(image))
    else:
        plt.imshow(np.array(image))
        # plt.imsave("supsup.jpg", np.array(image))

        plt.show()


if __name__ == "__main__":
    random.seed(42)

    config_path = "data/yolo/config.yaml"
    with open(config_path) as file:
        config = yaml.load(file, yaml.CLoader)

    class_id_to_name_mapping = {
        i: class_name for i, class_name in enumerate(config["names"])
    }
    class_name_to_id_mapping = {
        class_name: i for i, class_name in class_id_to_name_mapping.items()
    }

    COLORS = list(
        itertools.chain(
            *[
                ["orange", "green", "blue", "purple", "yellow", "brown"]
                for _ in range(8)
            ]
        )
    )

    print(class_id_to_name_mapping)
    print(class_name_to_id_mapping)

    # Get any random annotation file
    subdir, page, filename = "train", "page-1", "1__0@600__0@600.jpg"
    subdir, page, filename = "val", "page-3", "3__300@900__1200@1800.jpg"
    subdir, page, filename = "val", "page-5", "5__0800@1600__3600@4400.jpg"
    subdir, page, filename = "val", "page-6", "6__4400@5200__2400@3200.jpg"

    IMAGE_FILE = os.path.join(config[subdir], page, filename)
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
