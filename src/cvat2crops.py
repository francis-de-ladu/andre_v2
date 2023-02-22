import itertools
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import xmltodict
import yaml

from utils import Coords, Label


BLOCK_SIZE = 600
WIDTH, HEIGHT = 3600, 2400


def get_output_filename(page_id, start_x, start_y, block_size, *, extension):
    assert extension in ('jpg', 'txt')
    end_x, end_y = start_x + block_size, start_y + block_size
    return f"{page_id}__{start_x}@{end_x}__{start_y}@{end_y}.{extension}"


if __name__ == '__main__':
    DATA_DIR = Path("data")
    YOLO_DIR = DATA_DIR / "yolo"

    shutil.rmtree(YOLO_DIR, ignore_errors=True)

    xml_path = "./data/cvat/annotations.xml"
    with open(xml_path, 'r') as xmlfile:
        content = xmlfile.read()

    doc = xmltodict.parse(content)
    IMAGES_DATA = doc['annotations']['image']
    LABELS_DATA = doc['annotations']['meta']['task']['labels']['label']

    CLASSES = {infos['name']: Label(i, infos) for i, infos in enumerate(LABELS_DATA)}

    yolo_objects = []
    for image in IMAGES_DATA:
        img_id = image['@id']
        width = int(image['@width'])
        height = int(image['@height'])

        cvat_objects = []
        for c in CLASSES.values():
            if c.tag not in image:
                continue

            objs = image[c.tag]
            if isinstance(objs, list):
                cvat_objects.extend(objs)
            else:
                cvat_objects.append(objs)

        for obj in cvat_objects:
            label = CLASSES[obj['@label']]
            coords = Coords(obj, label.tag)
            yolo_objects.append(coords.to_yolo_crops(img_id, BLOCK_SIZE, label.cid))

    df = pd.DataFrame.from_dict(yolo_objects)

    print()
    print(f"{CLASSES = }")
    print()
    print(df)

    SUBDIRS = ("train", "val", "test")
    IMAGE_DIRS = {subdir: (YOLO_DIR / "images" / subdir) for subdir in SUBDIRS}
    LABEL_DIRS = {subdir: (YOLO_DIR / "labels" / subdir) for subdir in SUBDIRS}
    for subdir in itertools.chain(IMAGE_DIRS.values(), LABEL_DIRS.values()):
        subdir.mkdir(parents=True, exist_ok=True)

    with open("config/page_mapping.json", 'r') as config:
        page_mapping = json.load(config)

    for page_id, subdir in page_mapping.items():
        page_dir = f"page-{page_id}"
        src_path = DATA_DIR / 'images' / page_dir
        dst_path = IMAGE_DIRS[subdir] / page_dir
        shutil.copytree(src_path, dst_path)

        page_mask = (df.image_id == page_id)
        for start_x in range(0, WIDTH, BLOCK_SIZE):
            for start_y in range(0, HEIGHT, BLOCK_SIZE):
                crop_mask = (df.px == start_x // BLOCK_SIZE) & (df.py == start_y // BLOCK_SIZE)
                crop_annots = df.loc[page_mask & crop_mask].drop(['image_id', 'px', 'py'], axis=1)

                filename = get_output_filename(page_id, start_x, start_y, BLOCK_SIZE, extension='txt')
                annot_dir = LABEL_DIRS[subdir] / page_dir
                annot_dir.mkdir(parents=True, exist_ok=True)

                crop_annots.to_csv(annot_dir / filename, sep='\t', index=False, header=False)

    CONFIG_PATH = YOLO_DIR / "config.yaml"
    with open(CONFIG_PATH, 'w+') as file:
        dirs = {kind: f"../andre_v2/{path.as_posix()}" for kind, path in IMAGE_DIRS.items()}
        config = dict(**dirs, nc=len(CLASSES), names=list(CLASSES.keys()))
        file.write(yaml.dump(config))
