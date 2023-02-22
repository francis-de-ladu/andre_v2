#!/bin/bash

NAME=$1
PAGE=${2:-*}
CONF=${3:-.5}
CHECKPOINT=${4:-best}
IMG=${5:-800}

python ../yolov5/detect.py \
    --img ${IMG} \
    --name ${NAME} \
    --conf ${CONF} \
    --source "data/yolo/images/val/page-${PAGE}/*.jpg" \
    --weights ../yolov5/runs/train/${NAME}/weights/${CHECKPOINT}.pt \
    --hide-labels \
    --device cpu \
    --exist-ok \
    --line-thickness 2
