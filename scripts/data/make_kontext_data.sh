#!/bin/bash

DATA_ROOT="/workspace/codes/DiffSynth-Studio/data/neemo_mini_1440p_120f"
INPUT_IMAGE_DIR="${DATA_ROOT}/gt_images_1440p"
KONTEXT_IMAGE_DIR="${DATA_ROOT}/lq_images_720p"
OUTPUT_DIR="${DATA_ROOT}/kontext_data"
PROMPT="Make the image more detailed and realistic."

CUDA_VISIBLE_DEVICES=0 python3 scripts/data/make_kontext_data.py \
    --image_dir $INPUT_IMAGE_DIR \
    --kontext_image_dir $KONTEXT_IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --prompt "${PROMPT}"
