import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import pandas as pd
import random
from validate_annots import plot_bounding_box
from PIL import Image


def get_min_and(coords: str) -> tuple[int, int]:
    # val_min, val_max = map(int, coords.split("@"))
    # return (val_min, val_max) if val_min < val_max else (val_max, val_min)
    return map(int, coords.split("@"))


def get_start_and_size(coords: str) -> dict[str, int]:
    val_min, val_max = map(int, coords.split("@"))
    return dict(start=val_min, size=val_max - val_min)


def get_tile_params(tile_preds_path: Path) -> tuple[str, dict[str, dict[str, int]]]:
    pageid, xcoords, ycoords = tile_preds_path.stem.split("__")
    coords = dict(
        x=get_start_and_size(xcoords),
        y=get_start_and_size(ycoords),
    )
    return pageid, coords


def display_predictions(
    pageid: str, annots: pd.DataFrame, nb_colors: int = 2**24 - 1
) -> None:
    random.seed(42)
    colors = {
        int(clas): "#" + str(hex(random.randint(0, nb_colors))[2:].zfill(6))
        for clas in annots["cls"].unique()
    }

    image = Image.open(f"cvat/images/page-{pageid}.jpg")
    image = image.resize(map(lambda x: x * 2, image.size))

    plot_bounding_box(image, annots.values.tolist(), colors=colors)


def convert_to_bounding_boxes(annots: pd.DataFrame) -> pd.DataFrame:
    print(annots)
    annots["width"] = annots["x_size"]
    annots["height"] = annots["y_size"]
    annots["area"] = annots["width"] * annots["height"]

    annots["x0"] = annots["x_center"] - (annots["width"] / 2)
    annots["y0"] = annots["y_center"] - (annots["height"] / 2)
    annots["x1"] = annots["x_center"] + (annots["width"] / 2)
    annots["y1"] = annots["y_center"] + (annots["height"] / 2)
    return annots.drop(columns=["x_size", "y_size"], axis=1).round().astype(int)


def compute_overlap(rect1, rect2) -> float:
    overlap_x = max(rect1["x0"], rect2["x0"]) - min(rect1["x1"], rect2["x1"])
    overlap_y = max(rect1["y0"], rect2["y0"]) - min(rect1["y1"], rect2["y1"])
    if overlap_x > 0 or overlap_y > 0:
        return 0

    area_overlap = overlap_x * overlap_y
    # return area_overlap / rect1["area"]
    return area_overlap / (rect1["area"] + rect2["area"] - area_overlap)


def main(full_preds_path: Path) -> None:
    COLUMNS = ["cls", "x_center", "y_center", "x_size", "y_size"]
    all_preds = defaultdict(list)

    for tile_preds_path in sorted(full_preds_path.glob("*")):  # [:3]:
        pageid, coords = get_tile_params(tile_preds_path)

        with tile_preds_path.open("r") as fp:
            tile_preds = fp.read().splitlines()

        preds = pd.DataFrame(map(str.split, tile_preds), columns=COLUMNS)
        preds[COLUMNS[1:]] = preds[COLUMNS[1:]].astype(float)

        for col_name in COLUMNS[1:]:
            axis = coords[col_name[0]]
            denormalized = preds[col_name] * axis["size"]
            if "center" in col_name:
                denormalized += axis["start"]
            preds[col_name] = denormalized

        all_preds[pageid].extend(preds.values.tolist())

    all_preds = pd.DataFrame(all_preds[pageid], columns=COLUMNS)
    all_preds.sort_values(by=COLUMNS[1:], inplace=True)
    # display_predictions(pageid, all_preds.astype(int))

    all_preds = convert_to_bounding_boxes(all_preds)
    # all_preds["overlap"] = all_preds.iloc[:10].apply(
    all_preds["overlap"] = all_preds.apply(
        lambda x: compute_overlap(all_preds.iloc[0], x), axis=1
    )
    print(all_preds)


if __name__ == "__main__":
    full_preds_path = Path(sys.argv[1])
    main(full_preds_path)
