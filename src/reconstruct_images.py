import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import random
from validate_annots import plot_bounding_box
from PIL import Image
from tqdm import tqdm


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
    pageid: str,
    annots: pd.DataFrame,
    nb_colors: int = 2**24 - 1,
    save_only=False,
) -> None:
    random.seed(2023)
    colors = {
        int(clas): "#" + str(hex(random.randint(0, nb_colors))[2:].zfill(6))
        for clas in annots["cls"].unique()
    }

    image = Image.open(f"cvat/images/page-{pageid}.jpg")
    image = image.resize(map(lambda x: x * 2, image.size))

    plot_bounding_box(image, annots.values.tolist(), colors=colors, save_only=save_only)


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


def compute_overlap(rect1, rect2, symmetrical=False) -> float:
    overlap_x = max(rect1["x0"], rect2["x0"]) - min(rect1["x1"], rect2["x1"])
    overlap_y = max(rect1["y0"], rect2["y0"]) - min(rect1["y1"], rect2["y1"])
    if overlap_x > 0 or overlap_y > 0:
        return 0

    area_overlap = overlap_x * overlap_y
    if symmetrical:
        return area_overlap / (rect1["area"] + rect2["area"] - area_overlap)
    else:
        return area_overlap / rect1["area"]


def remove_overlaps(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    nb_preds = len(preds)
    overlap_ratios = np.ndarray([nb_preds, nb_preds])

    preds = preds.reset_index(drop=False)

    for idx1, pred1 in tqdm(preds.iterrows(), total=len(preds)):
        for idx2, pred2 in preds.iterrows():
            overlap_ratios[idx1, idx2] = compute_overlap(pred1, pred2, symmetrical=True)

    grouped_preds = set()
    for overlaps in overlap_ratios:
        indices = np.argwhere(overlaps > threshold).reshape(-1)
        grouped_preds.add(tuple(indices))

    # print(sorted(grouped_preds))

    cleaned_preds = []
    for indices in grouped_preds:
        current_preds = preds.loc[list(indices)].reset_index(drop=True)
        nb_current = len(current_preds)

        overlap_ratios = np.ndarray([nb_current, nb_current])
        for idx1, pred1 in current_preds.iterrows():
            for idx2, pred2 in current_preds.iterrows():
                overlap_ratios[idx1, idx2] = compute_overlap(
                    pred1, pred2, symmetrical=False
                )

        overall = overlap_ratios.sum(axis=0)
        best = current_preds.iloc[overall.argmax()]

        cleaned_preds.append(best)

    cleaned_preds = pd.DataFrame(cleaned_preds).drop_duplicates()
    return cleaned_preds.set_index("index")


def main(full_preds_path: Path, pageid: str | None) -> None:
    COLUMNS = ["cls", "x_center", "y_center", "x_size", "y_size"]
    all_preds = defaultdict(list)

    fn_format = "*.txt"
    if pageid is not None:
        fn_format = f"{pageid}__{fn_format}"

    for tile_preds_path in sorted(full_preds_path.glob(fn_format)):
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

    OUT_DIR = Path("out")
    OUT_DIR.mkdir(exist_ok=True)

    for pageid, raw_preds in all_preds.items():
        raw_preds = pd.DataFrame(raw_preds, columns=COLUMNS)

        formatted_preds = convert_to_bounding_boxes(raw_preds.copy())

        clean_preds = remove_overlaps(formatted_preds, threshold=0.2)
        clean_preds = remove_overlaps(clean_preds, threshold=0.5)

        save_preds = raw_preds.loc[clean_preds.index].round(1)
        save_preds.sort_values(
            by=["cls", "x_center", "y_center", "x_size", "y_size"], inplace=True
        )

        out_path = OUT_DIR / f"{pageid}.txt"
        save_preds.to_csv(out_path, header=False, index=False, sep="\t")

        # display_predictions(pageid, save_preds.astype(int))


if __name__ == "__main__":
    full_preds_path = Path(sys.argv[1])
    pageid = sys.argv[2] if len(sys.argv) > 2 else None

    if len(sys.argv) > 3:
        preds = pd.read_csv(
            full_preds_path,
            names=["cls", "x_center", "y_center", "x_size", "y_size"],
            index_col=False,
            sep="\t",
        )
        display_predictions(pageid, preds, save_only=True)
    else:
        main(full_preds_path, pageid)
