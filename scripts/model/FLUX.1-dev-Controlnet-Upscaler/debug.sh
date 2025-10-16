#! /bin/bash

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

DATASET_BASE_PATH="/workspace/codes/DiffSynth-Studio/data/neemo_mini_res_900/controlnet_data"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata_controlnet_upscale_data.csv"
OUTPUT_PATH="/workspace/codes/DiffSynth-Studio/outputs/train_lora_finetune/models/FLUX.1-dev-Controlnet-Upscaler_lora"
MAX_PIXELS=1440000 # 1600x900
IMG_HEIGHT=896
IMG_WIDTH=1600
DATASET_REPEAT=100
NUM_EPOCHS=5

accelerate launch --mixed_precision=bf16 examples/flux/model_training/train.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $DATASET_METADATA_PATH \
  --data_file_keys "image,controlnet_image" \
  --max_pixels $MAX_PIXELS \
  --height $IMG_HEIGHT \
  --width $IMG_WIDTH \
  --dataset_repeat $DATASET_REPEAT \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors,jasperai/Flux.1-dev-Controlnet-Upscaler:diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs $NUM_EPOCHS \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $OUTPUT_PATH \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --extra_inputs "controlnet_image" \
  --align_to_opensource_format \
  --use_gradient_checkpointing
