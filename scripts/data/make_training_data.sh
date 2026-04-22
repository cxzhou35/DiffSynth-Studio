#!/bin/bash

DATA_ROOT="data/siga_004_1_seq1"
INPUT_IMAGE_DIR="${DATA_ROOT}/cache/images_1440p"
COND_IMAGE_DIR="${DATA_ROOT}/cache/lr_data/lq_images_720p"
PROMPT="A young man in a maroon t-shirt and baggy beige pants strikes a dynamic pushing pose in a bright motion-capture studio filled with camera rigs and equipment."
PROMPT_DIR=""
COND_TYPES=("controlnet" "kontext")

for COND_TYPE in "${COND_TYPES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/data/prepare_metadata.py \
        --image_dir $INPUT_IMAGE_DIR \
        --cond_image_dir $COND_IMAGE_DIR \
        --output_dir "${DATA_ROOT}/${COND_TYPE}_data" \
        --prompt "${PROMPT}" \
        --prompt_dir "${PROMPT_DIR}" \
        --meta_type json \
        --cond_type ${COND_TYPE} \
        --split train \
        --view_range 0 28 1 \
        --remove_prefix "${DATA_ROOT}/"
done
