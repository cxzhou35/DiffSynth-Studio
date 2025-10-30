#! /bin/bash

export NUM_NODES=1
export NUM_GPUS=4

DATASET_BASE_PATH="data/neemo_mini_1440p_120f/controlnet_data"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata.csv"
OUTPUT_PATH="outputs/neemo_mini_1440p_120f/train_finetune_controlnet/models/FLUX.1-dev-Controlnet-Upscaler_controlnet"
# MAX_PIXELS=3686400 # 2560x1440
IMG_HEIGHT=1440
IMG_WIDTH=2560
DATASET_REPEAT=10
NUM_EPOCHS=5

#   --height $IMG_HEIGHT \
#   --width $IMG_WIDTH \
#   --max_pixels $MAX_PIXELS \

accelerate launch --mixed_precision=bf16 --multi_gpu --main_process_port 29501 --num_machines $NUM_NODES --num_processes $NUM_GPUS --config_file examples/flux/model_training/full/accelerate_config.yaml scripts/model/train.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $DATASET_METADATA_PATH \
  --data_file_keys "image,controlnet_images" \
  --dataset_repeat $DATASET_REPEAT \
  --height $IMG_HEIGHT \
  --width $IMG_WIDTH \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors,jasperai/Flux.1-dev-Controlnet-Upscaler:diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs $NUM_EPOCHS \
  --remove_prefix_in_ckpt "pipe.controlnet.models.0." \
  --output_path $OUTPUT_PATH \
  --trainable_models "controlnet" \
  --extra_inputs "controlnet_images" \
  --use_gradient_checkpointing \
  --find_unused_parameters
