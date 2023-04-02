import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
import sys

if __name__ == "__main__":
    pageid = sys.argv[1]

    COLUMNS = ["cls", "x_center", "y_center", "x_size", "y_size"]
    for filename in Path("out").glob(f"{pageid}.txt"):
        pageid = filename.stem
        data = pd.read_csv(filename, names=COLUMNS, sep="\t")

    labels_file = Path("config/label_mapping.json")
    labels_mapping = json.load(labels_file.open("r"))

    labels_dict = {i: obj["name"] for i, obj in enumerate(labels_mapping)}
    labels_dict |= {obj["id"]: obj["name"] for obj in labels_mapping}

    data["predicted"] = data["cls"].map(lambda cid: labels_dict.get(cid))
    predicted_counts = pd.DataFrame(data["predicted"].value_counts())

    annots_files = Path("cvat/annotations.json")
    objects = json.load(annots_files.open("r"))["shapes"]

    frame = int(pageid) - 1
    frame_objects = [obj for obj in objects if obj["frame"] == frame]

    real_counts = defaultdict(int)
    for obj in frame_objects:
        real_counts[labels_dict.get(obj["label_id"])] += 1

    real_counts = pd.DataFrame.from_dict(real_counts, orient="index", columns=["real"])

    predictions = pd.concat([real_counts, predicted_counts], axis=1)
    predictions = predictions.fillna(0).astype(int)

    predictions.sort_values(by="real", ascending=False, inplace=True)
    predictions["missing"] = predictions.real - predictions.predicted

    print(predictions)
