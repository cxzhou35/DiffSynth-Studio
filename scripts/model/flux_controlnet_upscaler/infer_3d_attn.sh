#! /bin/bash

# COND_IMAGE="/home/vercent/codes/DiffSynth-Studio/data/old_tim_1440p_240f/lq_images_720p_light_deg"
COND_IMAGE="/home/vercent/codes/minivolcap2/data/old_tim/cache/images_crop_lr"
PROMPT="A man wearing glasses is being filmed by a professional multi-camera rig in a brightly modern room."
HEIGHT=1440
WIDTH=2560
CKPT_PATH="/home/vercent/codes/DiffSynth-Studio/outputs/old_tim_1440p_240f_20251211080043/FLUX.1-Controlnet-Upscale-dit-lora_3d_attn/epoch-4-step-8700.safetensors"
OUTPUT_DIR="outputs/20251225_flux_controlnet_upscaler_lora_3d_attn_with_3d_rope_epoch_2/old_tim_crop_eval_0_119f"

# for loop to the DIR
CUDA_VISIBLE_DEVICES=0 python3 scripts/model/flux_controlnet_upscaler/infer_controlnet_3d_attn.py \
    --cond_image "$COND_IMAGE" \
    --prompt "${PROMPT}" \
    --height $HEIGHT \
    --width $WIDTH \
    --ckpt_path $CKPT_PATH \
    --output_dir "$OUTPUT_DIR" \
    --relative_frame_range 0 119 1 \
    --view_range 0 30 1 \
    --save_video \
    --use_3d_attn \
    --attn_window_size 4 \
    --use_3d_rope
