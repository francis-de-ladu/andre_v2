import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import pandas as pd
import requests
import torch
import torchvision
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import hdbscan


def delete_content(directory):
    for child in directory.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            delete_content(child)


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def load_segment_anything_model(model_type: str, checkpoint: str):
    model_path = Path("models") / checkpoint
    model_path.parent.mkdir(exist_ok=True)

    if model_path.exists():
        sam = sam_model_registry[model_type](checkpoint=model_path)
    else:
        url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint}"
        resp = requests.get(url, allow_redirects=True)
        model_path.write_bytes(resp.content)
        sam = sam_model_registry[model_type](checkpoint=checkpoint)

    return sam


def create_onnx_runtime_session(
    sam,
    onnx_model_path: str = "models/sam_onnx_example.onnx",
    quantized_model_path: str = "models/sam_onnx_quantized_example.onnx",
):
    export_onnx_model(sam, onnx_model_path)
    quantize_onnx_model(onnx_model_path, quantized_model_path)

    return onnxruntime.InferenceSession(quantized_model_path)


def export_onnx_model(sam, onnx_model_path: str) -> str:
    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(
            low=0, high=1024, size=(1, 5, 2), dtype=torch.float
        ),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )


def quantize_onnx_model(onnx_model_path: str, onnx_model_quantized_path: str) -> str:
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=onnx_model_quantized_path,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )


def get_input_params(image, step: int = 100) -> dict[str, np.ndarray]:
    input_points = []
    input_labels = []

    height, width = image.shape[:2]

    for x in range(step, width, step):
        for y in range(step, height, step):
            if x in (step, width - step) or y in (step, height - step):
                input_points.append([x, y])
                input_labels.append(0)

    for pos in range(1, 4):
        shift = step * ((pos - 1) % 2) * 4
        range_x = range(pos * width // 6, pos * width * 2 // 6 + 1, step)
        range_y = range(shift + height // 2, shift + height * 2 // 3 + 1, step)
        for i, (x, y) in enumerate(zip(range_x, range_y)):
            input_points.append([x, y])
            input_points.append([x, range_y.stop - i * step])
            input_labels.extend([1, 1])

    input_box = np.array([0, 0, width, height])
    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    return dict(box=input_box, points=input_points, labels=input_labels)


def extract_regions(predictor, idx, image, input_params: dict[str, np.ndarray]):
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image, cmap="gray")
    # # show_mask(masks[0], plt.gca())
    # # for box in areas:
    # #     show_box(box, plt.gca())
    # # show_points(input_point, input_label, plt.gca())
    # plt.axis("on")
    # plt.show()

    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    onnx_coord = np.concatenate(
        [input_params["points"], np.array([[0.0, 0.0]])], axis=0
    )[None, :, :]
    onnx_label = np.concatenate([input_params["labels"], np.array([-1])], axis=0)[
        None, :
    ].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(
        np.float32
    )

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32),
    }

    masks, _, _ = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    masks = np.where(
        masks.sum(keepdims=True, axis=-1) < 0.12 * masks.shape[-1], False, masks
    )
    masks = np.where(
        masks.sum(keepdims=True, axis=-2) < 0.12 * masks.shape[-2], False, masks
    )

    raw_coords = pd.DataFrame(np.argwhere(masks[0][0]), columns=["y", "x"]).iloc[::100]

    coords = StandardScaler().fit_transform(raw_coords)
    clustering = DBSCAN(eps=0.08, min_samples=10).fit(coords)
    # clustering = hdbscan.HDBSCAN(min_cluster_size=300).fit(coords)

    labels, counts = np.unique(clustering.labels_, return_counts=True)

    areas = []
    for label, count in zip(labels, counts):
        if count > 2500:
            points = raw_coords.iloc[clustering.labels_ == label]
            y_min, x_min = points.min(axis=0)
            y_max, x_max = points.max(axis=0)
            areas.append((x_min, y_min, x_max, y_max))

    print()
    print("Page", idx)
    print(sorted(counts, reverse=True))
    print("\n".join(map(str, areas)))

    if idx in (4, 6):
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray")
        show_mask(masks[0], plt.gca())
        for box in areas:
            show_box(box, plt.gca())
        show_points(input_params["points"], input_params["labels"], plt.gca())
        plt.axis("on")
        plt.show()

    new_masks = np.zeros_like(masks, dtype=bool)
    new_areas = []
    for area in areas:
        x_min, y_min, x_max, y_max = area
        area_mask = masks[:, :, y_min:y_max, x_min:x_max].copy()

        area_mask = np.where(
            area_mask.sum(keepdims=True, axis=-2) < 0.5 * area_mask.shape[-2],
            False,
            area_mask,
        )
        area_mask = np.where(
            area_mask.sum(keepdims=True, axis=-1) < 0.5 * area_mask.shape[-1],
            False,
            area_mask,
        )
        new_masks[:, :, y_min:y_max, x_min:x_max] = area_mask

        coords = np.argwhere(area_mask[0][0]) + [y_min, x_min]
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

        new_areas.append((x_min, y_min, x_max, y_max))

    if idx in (4, 6):
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray")
        show_mask(new_masks[0], plt.gca())
        for box in new_areas:
            show_box(box, plt.gca())
        show_points(input_params["points"], input_params["labels"], plt.gca())
        plt.axis("on")
        plt.show()

    return new_areas


if __name__ == "__main__":
    model_type = "vit_h"
    checkpoint = "sam_vit_h_4b8939.pth"

    DATA_DIR = Path("data_bis")
    DATA_DIR.mkdir(exist_ok=True)
    delete_content(DATA_DIR)

    sam = load_segment_anything_model(model_type, checkpoint)
    ort_session = create_onnx_runtime_session(sam)
    predictor = SamPredictor(sam.to(device="cuda"))

    input_params = None
    for idx in range(1, 8):
        image = cv2.imread(f"cvat/images/page-{idx}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_size = image.shape[0] * image.shape[1]

        if input_params is None:
            input_params = get_input_params(image, step=100)

        regions = extract_regions(predictor, idx, image, input_params)
        regions = pd.DataFrame(regions, columns=["x_min", "y_min", "x_max", "y_max"])

        regions["width"] = regions.x_max - regions.x_min
        regions["height"] = regions.y_max - regions.y_min
        regions["area"] = regions.width * regions.height

        regions = regions.loc[regions.area < 0.8 * image_size]
        regions.sort_values("area", ascending=False, inplace=True)

        for i, (__, reg) in enumerate(regions.iterrows(), start=1):
            region_path = DATA_DIR / f"page-{idx}-{i}.jpg"
            region_image = image[reg.y_min : reg.y_max, reg.x_min : reg.x_max]
            cv2.imwrite(region_path.as_posix(), region_image)
