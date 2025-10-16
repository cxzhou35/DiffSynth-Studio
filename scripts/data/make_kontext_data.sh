#!/bin/bash

INPUT_IMAGE_DIR="/workspace/data/neemo_mini/deg_2d/gt"
KONTEXT_IMAGE_DIR="/workspace/data/neemo_mini/deg_2d/render"
OUTPUT_DIR="/workspace/codes/DiffSynth-Studio/data/neemo_mini_res_900/kontext_data"
PROMPT="In the foreground, the artist is performing with focused intensity, her sleek black hair and white sleeveless top complemented by a striking silver necklace, She holds a microphone while singing under a bright spotlight, The background reveals professional photography equipment, including light stands and cameras, with visible cables on the floor and a dark backdrop, The stage atmosphere is intimate and polished, emphasizing her elegance and the quiet concentration of the scene."

CUDA_VISIBLE_DEVICES=0 python3 make_kontext_data.py \
    --image_dir $INPUT_IMAGE_DIR \
    --kontext_image_dir $KONTEXT_IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --prompt "${PROMPT}"
