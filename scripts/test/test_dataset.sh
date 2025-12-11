#! /bin/bash

# get time now
SCENE_ID="old_tim_1440p_240f"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
DATASET_BASE_PATH="data/${SCENE_ID}"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/controlnet_data/metadata_train.json"
IMG_HEIGHT=1440
IMG_WIDTH=2560

python3 scripts/test/test_mvdataset.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $DATASET_METADATA_PATH \
  --data_file_keys "image,controlnet_images" \
  --height $IMG_HEIGHT \
  --width $IMG_WIDTH \
