#! /bin/bash
export NUM_NODES=1
export NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES="0"

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export ACCELERATE_DEBUG_MODE="1"

DATASET_BASE_PATH="data/old_tim_1440p_120f/kontext_data"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata.json"
OUTPUT_PATH="outputs/debug"
# MAX_PIXELS=3686400 # 2560x1440
IMG_HEIGHT=1440
IMG_WIDTH=2560
DATASET_REPEAT=10
NUM_EPOCHS=1

accelerate launch --mixed_precision=bf16 --num_processes 1 scripts/models/train.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $DATASET_METADATA_PATH \
  --data_file_keys "image,kontext_images" \
  --height $IMG_HEIGHT \
  --width $IMG_WIDTH \
  --dataset_repeat $DATASET_REPEAT \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-Kontext-dev:flux1-kontext-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs $NUM_EPOCHS \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $OUTPUT_PATH \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --align_to_opensource_format \
  --extra_inputs "kontext_images" \
  --use_gradient_checkpointing
