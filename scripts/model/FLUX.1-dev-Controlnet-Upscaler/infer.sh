#! /bin/bash

# CONTROL_IMAGE="/workspace/codes/DiffSynth-Studio/data/neemo_mini_res_900/controlnet_data/render/32"
CONTROL_IMAGE="/workspace/codes/DiffSynth-Studio/data/old_tim_1440p_120f/lq_images_720p"
DIR=(43)

PROMPT="In a brightly lit, modern room, an East Asian man wearing glasses and a grey hoodie speaks with a thoughtful expression, his hands clasped in front of him. The setting suggests a professional interview or video shoot, with a multi-camera rig visible on the right, contrasted by a comfortable environment featuring a large potted plant, a brick wall, and a large window looking out onto a sunny, green yard. The overall atmosphere is calm and focused, blending a professional production with a relaxed, naturalistic setting."
HEIGHT=1440
WIDTH=2560
CKPT_PATH="/workspace/codes/DiffSynth-Studio/outputs/old_tim_1440p_120f/train_lora_finetune/models/FLUX.1-dev-Controlnet-Upscaler_lora/epoch-0.safetensors"
OUTPUT_DIR="/workspace/codes/DiffSynth-Studio/outputs/old_tim_1440p_120f/train_lora_finetune/inference/train_data_eval"

# for loop to the DIR
for d in "${DIR[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/models/FLUX.1-dev-Controlnet-Upscaler/infer.py \
        --control_image "$CONTROL_IMAGE/$d" \
        --prompt "${PROMPT}" \
        --height $HEIGHT \
        --width $WIDTH \
        --ckpt_path $CKPT_PATH \
        --output_dir "$OUTPUT_DIR/$d"
done
