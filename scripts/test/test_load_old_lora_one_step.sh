#! /bin/bash
set -euo pipefail

export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS=${NUM_GPUS:-1}

TEST_ID="load_old_lora_one_step"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
DATASET_BASE_PATH="/home/vercent/codes/minivolcap2"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/data/siga_all_1440p_lq720p/test_one_step_controlnet_data/metadata_train.json"
PRETRAINED_MODEL_PATH="/home/vercent/codes/DiffSynth-Studio/outputs/siga_004_1_seq1_20260422154531/FLUX.1-Controlnet-Upscale-dit-lora_3d_attn_3d_rope/epoch-0-step-750.safetensors"
OUTPUT_PATH="outputs/${TEST_ID}_${TIMESTAMP}/FLUX.1-Controlnet-Upscale-dit-lora_checkpoint_test"
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29603}

if [[ ! -f "${DATASET_METADATA_PATH}" ]]; then
    echo "Missing dataset metadata: ${DATASET_METADATA_PATH}" >&2
    exit 1
fi
if [[ ! -f "${PRETRAINED_MODEL_PATH}" ]]; then
    echo "Missing pretrained checkpoint: ${PRETRAINED_MODEL_PATH}" >&2
    exit 1
fi

accelerate launch \
  --mixed_precision=bf16 \
  --main_process_port ${MAIN_PROCESS_PORT} \
  --num_machines ${NUM_NODES} \
  --num_processes ${NUM_GPUS} \
  scripts/model/train.py \
  --dataset_base_path ${DATASET_BASE_PATH} \
  --dataset_metadata_path ${DATASET_METADATA_PATH} \
  --data_file_keys "image,controlnet_images" \
  --max_pixels 262144 \
  --height 512 \
  --width 512 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors,jasperai/Flux.1-dev-Controlnet-Upscaler:diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --save_steps 1 \
  --dataset_num_workers 0 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path ${OUTPUT_PATH} \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --extra_inputs "controlnet_images" \
  --align_to_opensource_format \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --dit_3d_attn_interval 3 \
  --project_name "training" \
  --pretrained_model_path ${PRETRAINED_MODEL_PATH}
