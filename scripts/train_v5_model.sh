#!/bin/bash

MODEL=$1
EPOCHS=${2:-2000}
PATIENCE=${3:-0}
WORKERS=${4:-4}
IMG=${5:-600}

NAME=${MODEL}_${IMG}

python ../yolov5/train.py \
    --img ${IMG} \
    --cfg ${MODEL}.yaml \
    --weights ${MODEL}.pt \
    --workers ${WORKERS} \
    --epochs ${EPOCHS} \
    --data data/yolo/config.yaml \
    --name andre \
    --batch -1 \
    --cache ram
