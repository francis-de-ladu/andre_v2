import itertools
import json
import os
import shutil
import pandas as pd
import xmltodict
import yaml

from utils import Coords, Label


if __name__ == '__main__':
    CVAT_DIR = "./data/cvat/"
    YOLO_DIR = "../andre_v2/data/yolo/"

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
            yolo_objects.append(coords.to_yolo_object(img_id, width, height, label.cid))

    df = pd.DataFrame.from_dict(yolo_objects)

    print()
    print(f"{CLASSES = }")
    print()
    print(df)

    SUBDIRS = ("train", "val", "test")
    IMAGE_DIRS = {subdir: os.path.join(YOLO_DIR, "images", subdir) for subdir in SUBDIRS}
    LABEL_DIRS = {subdir: os.path.join(YOLO_DIR, "labels", subdir) for subdir in SUBDIRS}
    for subdir in itertools.chain(IMAGE_DIRS.values(), LABEL_DIRS.values()):
        os.makedirs(subdir, exist_ok=True)

    with open("config/page_mapping.json", 'r') as config:
        page_mapping = json.load(config)

    CVAT_IMAGES = os.path.join(CVAT_DIR, "images")
    for page_id, subdir in page_mapping.items():
        page_data = df.loc[df.image_id == page_id].drop('image_id', axis=1)
        image_fn = f"page-{page_id}.jpg"

        annot_path = os.path.join(LABEL_DIRS[subdir], f"page-{page_id}.txt")
        page_data.to_csv(annot_path, sep='\t', index=False, header=False)
        shutil.copy(os.path.join(CVAT_IMAGES, image_fn), os.path.join(IMAGE_DIRS[subdir], image_fn))

    CONFIG_PATH = os.path.join(YOLO_DIR, "config.yaml")
    with open(CONFIG_PATH, 'w+') as file:
        config = dict(
            **IMAGE_DIRS,
            nc=len(CLASSES),
            names=list(CLASSES.keys()),
        )
        file.write(yaml.dump(config))
