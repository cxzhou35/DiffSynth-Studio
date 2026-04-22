#! /bin/bash

# COND_IMAGE="/home/vercent/codes/DiffSynth-Studio/data/old_tim_1440p_240f/lq_images_720p_light_deg"
# COND_IMAGE="/home/vercent/codes/minivolcap2/data/new_dance_color/cache/images_crop_lr"
# COND_IMAGE="/home/vercent/codes/minivolcap2/evals/0114_old_tim_orig_sf_2k_res_render/val/images/pred"
# COND_IMAGE="/home/vercent/codes/minivolcap2/data/liuhaocun_data/lhc_lr_test/s4_tk3/cache/images_crop_lr"
COND_IMAGE="/home/vercent/codes/DiffSynth-Studio/data/old_tim_1440p_120f/gs_images_render"
# PROMPT="An man wearing glasses is being filmed by a professional multi-camera rig in a brightly modern room."
PROMPT="A young black-haired woman wearing a plaid bandanna with a blue logo looks at the camera and makes a V sign in a casually lit indoor setting."
HEIGHT=1440
WIDTH=2560
CKPT_PATH="/home/vercent/codes/DiffSynth-Studio/outputs/old_tim_1440p_120f_20260401012534/FLUX.1-Controlnet-Upscale-dit-lora_3d_attn_3d_rope_gs_fix/epoch-1-step-1740.safetensors"
OUTPUT_DIR="outputs/20260402_old_tim_1440p_120f_gs_render_3d_attn_infer"

# for loop to the DIR
CUDA_VISIBLE_DEVICES=1 python3 scripts/model/flux_controlnet_upscaler/infer_controlnet_3d_attn.py \
    --cond_image "$COND_IMAGE" \
    --prompt "${PROMPT}" \
    --height $HEIGHT \
    --width $WIDTH \
    --ckpt_path $CKPT_PATH \
    --output_dir "$OUTPUT_DIR" \
    --view_range 2 3 1 \
    --relative_frame_range 0 29 1 \
    --save_video \
    --use_3d_attn \
    --attn_window_size 4 \
    --use_3d_rope
