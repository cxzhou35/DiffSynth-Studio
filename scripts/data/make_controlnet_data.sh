#!/bin/bash

DATA_ROOT="/workspace/codes/DiffSynth-Studio/data/neemo_mini_1440p_120f"
INPUT_IMAGE_DIR="${DATA_ROOT}/gt_images_1440p"
CONTROLNET_IMAGE_DIR="${DATA_ROOT}/lq_images_720p"
OUTPUT_DIR="${DATA_ROOT}/controlnet_data"
PROMPT="In the foreground, the artist is performing with focused intensity, her sleek black hair and white sleeveless top complemented by a striking silver necklace, She holds a microphone while singing under a bright spotlight, The background reveals professional photography equipment, including light stands and cameras, with visible cables on the floor and a dark backdrop, The stage atmosphere is intimate and polished, emphasizing her elegance and the quiet concentration of the scene."

CUDA_VISIBLE_DEVICES=0 python3 scripts/data/make_controlnet_data.py \
    --image_dir $INPUT_IMAGE_DIR \
    --controlnet_image_dir $CONTROLNET_IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --prompt "${PROMPT}"
