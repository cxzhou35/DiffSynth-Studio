#! /bin/bash

# get time now
SCENE_ID="neemo_mini_1440p_120f"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
DATASET_BASE_PATH="data/${SCENE_ID}/kontext_data"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata.json"
IMG_HEIGHT=1440
IMG_WIDTH=2560
NUM_EPOCHS=5

python3 scripts/test_dataset.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $DATASET_METADATA_PATH \
  --data_file_keys "image,kontext_images" \
  --height $IMG_HEIGHT \
  --width $IMG_WIDTH \
