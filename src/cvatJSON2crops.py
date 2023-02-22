import itertools
import json
import os
import shutil
from pathlib import Path
import math
import pandas as pd
import yaml

from utils import Coords, Label, Coords2


BLOCK_SIZE = 600
HALF_BLOCK = BLOCK_SIZE // 2

WIDTH, HEIGHT = 3600, 2400


def get_related_frame(all_frames, shape):
    fid = shape["frame"]
    return all_frames[fid]


def get_frame_size(frame):
    return frame["width"], frame["height"]


def get_output_filename(page_id, start_x, start_y, end_x, end_y):
    return f"{page_id}__{start_x}@{end_x}__{start_y}@{end_y}.txt"


if __name__ == "__main__":
    CVAT_DIR = Path("cvat")
    DATA_DIR = Path("data")
    YOLO_DIR = DATA_DIR / "yolo"
    CONFIG_DIR = Path("config")

    shutil.rmtree(YOLO_DIR, ignore_errors=True)

    with (
        open(CVAT_DIR / "meta.json", "r") as meta_file,
        open(CVAT_DIR / "annotations.json", "r") as annots_file,
        open(CONFIG_DIR / "label_mapping.json", "r") as labels_file,
    ):
        meta = json.load(meta_file)
        labels = json.load(labels_file)
        annotations = json.load(annots_file)

    all_frames = meta["frames"]
    all_shapes = annotations["shapes"]

    raw_annotations = pd.DataFrame.from_dict(
        [Coords2(shape).to_dict() for shape in all_shapes]
    )

    labels_mapping = {label["id"]: idx for idx, label in enumerate(labels)}
    raw_annotations["label"] = raw_annotations["label"].apply(
        lambda lab: labels_mapping.get(lab)
    )

    print(raw_annotations)

    SUBDIRS = ("train", "val", "test")
    IMAGE_DIRS = {subdir: (YOLO_DIR / "images" / subdir) for subdir in SUBDIRS}
    LABEL_DIRS = {subdir: (YOLO_DIR / "labels" / subdir) for subdir in SUBDIRS}
    for subdir in itertools.chain(IMAGE_DIRS.values(), LABEL_DIRS.values()):
        subdir.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_DIR / "page_mapping.json", "r") as config:
        page_mapping = json.load(config)

    yolo_annotations = []
    for page_id, subdir in page_mapping.items():
        page_dir = f"page-{page_id}"
        src_path = DATA_DIR / "images" / page_dir
        dst_path = IMAGE_DIRS[subdir] / page_dir
        shutil.copytree(src_path, dst_path)

        annots_dir = LABEL_DIRS[subdir] / page_dir
        annots_dir.mkdir(parents=True, exist_ok=True)

        page_mask = raw_annotations.frame == page_id
        for start_x in range(0, WIDTH, HALF_BLOCK):
            end_x = start_x + BLOCK_SIZE
            x_mask = raw_annotations.cx.between(start_x, end_x, inclusive="left")
            for start_y in range(0, HEIGHT, HALF_BLOCK):
                end_y = start_y + BLOCK_SIZE
                y_mask = raw_annotations.cy.between(start_y, end_y, inclusive="left")

                crop_annotations = raw_annotations.loc[
                    page_mask & x_mask & y_mask
                ].drop("frame", axis=1)

                crop_annotations["cx"] = (crop_annotations["cx"] - start_x) / BLOCK_SIZE
                crop_annotations["cy"] = (crop_annotations["cy"] - start_y) / BLOCK_SIZE
                crop_annotations["w"] /= BLOCK_SIZE
                crop_annotations["h"] /= BLOCK_SIZE

                rounded_cols = ["cx", "cy", "w", "h"]
                crop_annotations[rounded_cols] = crop_annotations[rounded_cols].round(5)

                filename = get_output_filename(page_id, start_x, start_y, end_x, end_y)
                crop_annotations.to_csv(
                    annots_dir / filename, sep="\t", index=False, header=False
                )

    CONFIG_PATH = YOLO_DIR / "config.yaml"
    with open(CONFIG_PATH, "w+") as file:
        dirs = {kind: f"../andre_v2/{path}" for kind, path in IMAGE_DIRS.items()}
        config = dict(**dirs, nc=len(labels), names=[lab["name"] for lab in labels])
        file.write(yaml.dump(config))
