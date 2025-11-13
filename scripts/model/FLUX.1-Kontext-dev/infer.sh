#! /bin/bash

CONTROL_IMAGE="/home/vercent/codes/DiffSynth-Studio/data/old_tim_1440p_120f/lq_images_720p"
DIR=($(seq -f "%02g" 32 45))

PROMPT="in a brightly lit, modern room, an east asian man wearing glasses and a grey hoodie speaks with a thoughtful expression, his hands clasped in front of him. the setting suggests a professional interview or video shoot, with a multi-camera rig visible on the right, contrasted by a comfortable environment featuring a large potted plant, a brick wall, and a large window looking out onto a sunny, green yard. the overall atmosphere is calm and focused, blending a professional production with a relaxed, naturalistic setting."
HEIGHT=1440
WIDTH=2560
CKPT_PATH="/home/vercent/codes/DiffSynth-Studio/outputs/epoch-1.safetensors"
OUTPUT_DIR="outputs/old_tim_1440p_120f/train_lora_ft_with_3d_attn/inference/train_data_eval"

# for loop to the DIR
for d in "${DIR[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/model/infer_kontext.py \
        --control_image "$CONTROL_IMAGE/$d" \
        --prompt "${PROMPT}" \
        --height $HEIGHT \
        --width $WIDTH \
        --ckpt_path $CKPT_PATH \
        --output_dir "$OUTPUT_DIR/$d" \
        --save_video \
        --video_path "$OUTPUT_DIR/videos_60fps/$d.mp4"
done
