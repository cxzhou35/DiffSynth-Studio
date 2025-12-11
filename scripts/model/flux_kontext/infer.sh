#! /bin/bash

# COND_IMAGE="/home/vercent/codes/DiffSynth-Studio/data/old_tim_1440p_240f/lq_images_720p_resize"
COND_IMAGE="/home/vercent/codes/minivolcap2/data/old_tim/cache/images_crop_lr/"
DIR=($(seq -f "%02g" 6 7))

# PROMPT="Enhance image clarity and resolution while keeping the content identical. super-resolution, high detail, 4K clarity, same composition, natural texture."
PROMPT="A man wearing glasses is being filmed by a professional multi-camera rig in a brightly modern room."
HEIGHT=1440
WIDTH=2560
CKPT_PATH="/home/vercent/codes/DiffSynth-Studio/outputs/old_tim_1440p_240f_20251126060437/FLUX.1-Kontext-dev-lora-wo_3d_attn_deg_resize/epoch-1-step-6960.safetensors"
OUTPUT_DIR="outputs/old_tim_1440p_crop_sr/FLUX.1-Kontext-dev-lora-wo_3d_attn_deg_resize/inference/train_data_eval_new"

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
