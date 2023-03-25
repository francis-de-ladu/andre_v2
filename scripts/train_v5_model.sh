#!/bin/bash

MODEL=$1
EPOCHS=${2:-2000}
PATIENCE=${3:-0}
WORKERS=${4:-4}
IMG=${5:-800}

NAME=${MODEL}_${IMG}_
# --weights ${MODEL}.pt \
python ../yolov5/train.py \
    --img ${IMG} \
    --cfg ${MODEL}.yaml \
    --weights ../yolov5/runs/train/yolov5x_800_/weights/last.pt \
    --workers ${WORKERS} \
    --epochs ${EPOCHS} \
    --name ${NAME} \
    --data data/yolo/config.yaml \
    --hyp data/yolo/hyp.yaml \
    --batch -1 \
    --cache ram
