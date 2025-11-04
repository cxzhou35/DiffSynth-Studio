#!/bin/bash

DATA_ROOT="data/old_tim_1440p_120f"
INPUT_IMAGE_DIR="${DATA_ROOT}/gt_images_1440p"
COND_IMAGE_DIR="${DATA_ROOT}/lq_images_720p"
KONTEXT_IMAGE_DIR="${DATA_ROOT}/lq_images_720p"
PROMPT="An East Asian man wearing glasses is being filmed by a professional multi-camera rig for an interview in a brightly modern room."
COND_TYPES=("controlnet" "kontext")

for COND_TYPE in "${COND_TYPES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/data/prepare_metadata.py \
        --image_dir $INPUT_IMAGE_DIR \
        --cond_image_dir $COND_IMAGE_DIR \
        --output_dir ${DATA_ROOT}/${COND_TYPE}_data \
        --prompt "${PROMPT}" \
        --meta_type csv \
        --cond_type ${COND_TYPE}
done
