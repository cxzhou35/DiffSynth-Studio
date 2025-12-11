#!/bin/bash

DATA_ROOT="data/old_tim_1440p_240f"
INPUT_IMAGE_DIR="${DATA_ROOT}/gt_images_1440p"
COND_IMAGE_DIR="${DATA_ROOT}/lq_images_720p_light_deg"
PROMPT="A man wearing glasses is being filmed by a professional multi-camera rig in a brightly modern room."
COND_TYPES=("controlnet" "kontext")

for COND_TYPE in "${COND_TYPES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/data/prepare_metadata.py \
        --image_dir $INPUT_IMAGE_DIR \
        --cond_image_dir $COND_IMAGE_DIR \
        --output_dir "${DATA_ROOT}/${COND_TYPE}_data" \
        --prompt "${PROMPT}" \
        --meta_type json \
        --cond_type ${COND_TYPE} \
        --split train \
        --frame_range 0 240 1 \
        --view_range 32 60 1 \
        --remove_prefix "${DATA_ROOT}/"
done
