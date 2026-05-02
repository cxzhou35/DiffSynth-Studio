#! /bin/bash
set -euo pipefail

export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS=${NUM_GPUS:-4}

SCENE_ID="siga_all_scenes"
COND_TYPE="${COND_TYPE:-controlnet}"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
DATASET_BASE_PATH="/home/vercent/codes/minivolcap2"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/data/siga_all_1440p_lq720p/${COND_TYPE}_data/metadata_train.json"
OUTPUT_PATH="outputs/${SCENE_ID}_${COND_TYPE}_mixed_${TIMESTAMP}/FLUX.1-Controlnet-Upscale-dit-lora_3d_attn_3d_rope"
MAX_PIXELS=${MAX_PIXELS:-921600}
IMG_HEIGHT=${IMG_HEIGHT:-1440}
IMG_WIDTH=${IMG_WIDTH:-2560}
DATASET_REPEAT=${DATASET_REPEAT:-2}
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}
TEMPORAL_WINDOW_SIZE=${TEMPORAL_WINDOW_SIZE:-4}
SPATIAL_WINDOW_SIZE=${SPATIAL_WINDOW_SIZE:-4}
JOINT_TEMPORAL_WINDOW_SIZE=${JOINT_TEMPORAL_WINDOW_SIZE:-2}
JOINT_SPATIAL_WINDOW_SIZE=${JOINT_SPATIAL_WINDOW_SIZE:-2}
MIXED_TEMPORAL_PROB=${MIXED_TEMPORAL_PROB:-0.4}
MIXED_SPATIAL_PROB=${MIXED_SPATIAL_PROB:-0.4}
MIXED_JOINT_PROB=${MIXED_JOINT_PROB:-0.2}

if [[ ! -f "${DATASET_METADATA_PATH}" ]]; then
    echo "Missing dataset metadata: ${DATASET_METADATA_PATH}" >&2
    exit 1
fi

accelerate launch \
  --mixed_precision=bf16 \
  --multi_gpu \
  --main_process_port ${MAIN_PROCESS_PORT} \
  --num_machines ${NUM_NODES} \
  --num_processes ${NUM_GPUS} \
  scripts/model/train.py \
  --dataset_base_path ${DATASET_BASE_PATH} \
  --dataset_metadata_path ${DATASET_METADATA_PATH} \
  --data_file_keys "image,${COND_TYPE}_images" \
  --max_pixels ${MAX_PIXELS} \
  --height ${IMG_HEIGHT} \
  --width ${IMG_WIDTH} \
  --dataset_repeat ${DATASET_REPEAT} \
  --use_temporal_sample \
  --use_spatial_sample \
  --sample_mode mixed \
  --temporal_window_size ${TEMPORAL_WINDOW_SIZE} \
  --spatial_window_size ${SPATIAL_WINDOW_SIZE} \
  --joint_temporal_window_size ${JOINT_TEMPORAL_WINDOW_SIZE} \
  --joint_spatial_window_size ${JOINT_SPATIAL_WINDOW_SIZE} \
  --mixed_sampling_probs ${MIXED_TEMPORAL_PROB} ${MIXED_SPATIAL_PROB} ${MIXED_JOINT_PROB} \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors,jasperai/Flux.1-dev-Controlnet-Upscaler:diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path ${OUTPUT_PATH} \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --extra_inputs "${COND_TYPE}_images" \
  --align_to_opensource_format \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --dit_3d_attn_interval 3 \
  --use_3d_rope \
  --project_name "training"
