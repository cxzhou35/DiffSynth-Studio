#! /bin/bash

COND_IMAGE="/workspace/codes/minivolcap2/data/old_tim/cache/images_crop_lr_720p"
DIR=($(seq -f "%02g" 04 16))

PROMPT="In a brightly lit, modern room, an East Asian man wearing glasses and a grey hoodie speaks with a thoughtful expression, his hands clasped in front of him. The setting suggests a professional interview or video shoot, with a multi-camera rig visible on the right, contrasted by a comfortable environment featuring a large potted plant, a brick wall, and a large window looking out onto a sunny, green yard. The overall atmosphere is calm and focused, blending a professional production with a relaxed, naturalistic setting."
HEIGHT=1440
WIDTH=2560
CKPT_PATH="outputs/old_tim_1440p_120f/train_lora_finetune/models/FLUX.1-dev-Controlnet-Upscaler_lora/epoch-4.safetensors"
OUTPUT_DIR="outputs/old_tim_1440p_180f/train_lora_finetune/inference/test_data_eval"

# for loop to the DIR
for d in "${DIR[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/model/FLUX.1-dev-Controlnet-Upscaler/infer_controlnet.py \
        --control_image "$COND_IMAGE/$d" \
        --prompt "${PROMPT}" \
        --height $HEIGHT \
        --width $WIDTH \
        --ckpt_path $CKPT_PATH \
        --output_dir "$OUTPUT_DIR/$d" \
        --save_video \
        --video_path "$OUTPUT_DIR/videos_180f_60fps/$d.mp4"
done
