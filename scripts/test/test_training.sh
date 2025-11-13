#! /bin/bash
export NUM_NODES=1
export NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES="0"

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export ACCELERATE_DEBUG_MODE="1"

SCENE_ID="old_tim_1440p_120f"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
DATASET_BASE_PATH="data/${SCENE_ID}"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/kontext_data/metadata_train.json"
OUTPUT_PATH="outputs/debug_${SCENE_ID}_${TIMESTAMP}"
MAX_PIXELS=921600 # 1280x720 for kontext images
IMG_HEIGHT=1440
IMG_WIDTH=2560
DATASET_REPEAT=1
NUM_EPOCHS=1
#   --use_temporal_sample \
#   --temporal_window_size 4 \
#   --lora_rank 32 \

accelerate launch --mixed_precision=bf16 scripts/model/train.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $DATASET_METADATA_PATH \
  --data_file_keys "image,kontext_images" \
  --max_pixels $MAX_PIXELS \
  --height $IMG_HEIGHT \
  --width $IMG_WIDTH \
  --dataset_repeat $DATASET_REPEAT \
  --use_temporal_sample \
  --temporal_window_size 3 \
  --kontext_ref_offsets 1 0 0 \
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
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --dit_3d_attn_interval 3 \
