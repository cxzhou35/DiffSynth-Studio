#!/bin/bash

INPUT_IMAGE_DIR="/workspace/codes/DiffSynth-Studio/data/old_tim_1440p_120f/gt_images_1440p"
CONTROLNET_IMAGE_DIR="/workspace/codes/DiffSynth-Studio/data/old_tim_1440p_120f/lq_images_720p"
OUTPUT_DIR="/workspace/codes/DiffSynth-Studio/data/old_tim_1440p_120f/controlnet_data"
PROMPT="In a brightly lit, modern room, an East Asian man wearing glasses and a grey hoodie speaks with a thoughtful expression, his hands clasped in front of him. The setting suggests a professional interview or video shoot, with a multi-camera rig visible on the right, contrasted by a comfortable environment featuring a large potted plant, a brick wall, and a large window looking out onto a sunny, green yard. The overall atmosphere is calm and focused, blending a professional production with a relaxed, naturalistic setting.
"

CUDA_VISIBLE_DEVICES=0 python3 make_controlnet_data.py \
    --image_dir $INPUT_IMAGE_DIR \
    --controlnet_image_dir $CONTROLNET_IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --prompt "${PROMPT}"
