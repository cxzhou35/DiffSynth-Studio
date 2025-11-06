#! /bin/bash

export NUM_NODES=1
export NUM_GPUS=4

# get time now
SCENE_ID="neemo_mini_1440p_120f"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
DATASET_BASE_PATH="data/${SCENE_ID}"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata.json"
OUTPUT_PATH="outputs/${SCENE_ID}_${TIMESTAMP}/FLUX.1-Kontext-dev-lora/models"
# MAX_PIXELS=3686400 # 2560x1440
IMG_HEIGHT=1440
IMG_WIDTH=2560
DATASET_REPEAT=2
NUM_EPOCHS=5

accelerate launch --mixed_precision=bf16 --multi_gpu --main_process_port 29501 --num_machines $NUM_NODES --num_processes $NUM_GPUS scripts/model/train.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $DATASET_METADATA_PATH \
  --data_file_keys "image,kontext_images" \
  --height $IMG_HEIGHT \
  --width $IMG_WIDTH \
  --dataset_repeat $DATASET_REPEAT \
  --use_temporal_sample \
  --temporal_window_size 4 \
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
