#! /bin/bash

COND_IMAGE="/home/vercent/codes/DiffSynth-Studio/data/old_tim_1440p_120f/lq_images_720p"
DIR=($(seq -f "%02g" 32 45))

PROMPT="An East Asian man wearing glasses is being filmed by a professional multi-camera rig for an interview in a brightly modern room."
HEIGHT=1440
WIDTH=2560
CKPT_PATH="/home/vercent/codes/DiffSynth-Studio/outputs/epoch-1.safetensors"
OUTPUT_DIR="outputs/old_tim_1440p_120f/train_lora_ft_with_3d_attn/inference/train_data_eval"

# for loop for the DIR
for d in "${DIR[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/model/flux_kontext/infer_kontext.py \
        --cond_image "$COND_IMAGE/$d" \
        --prompt "${PROMPT}" \
        --height $HEIGHT \
        --width $WIDTH \
        --ckpt_path $CKPT_PATH \
        --output_dir "$OUTPUT_DIR/$d" \
        --save_video \
        --video_path "$OUTPUT_DIR/videos_60fps/$d.mp4"
done
