#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---------- User Config ----------
COND_IMAGE="/home/vercent/codes/DiffSynth-Studio/data/siga_004_1_seq1/cache/lr_data/lq_images_720p"
PROMPT="A young man in a maroon t-shirt and baggy beige pants strikes a dynamic pushing pose in a bright motion-capture studio filled with camera rigs and equipment."
HEIGHT=1440
WIDTH=2560
CKPT_PATH="/home/vercent/codes/DiffSynth-Studio/outputs/siga_004_1_seq1_20260422154531/FLUX.1-Controlnet-Upscale-dit-lora_3d_attn_3d_rope/epoch-0-step-750.safetensors"
OUTPUT_DIR="outputs/20260423_siga_004_1_seq1_lq720p_v25_28_29_epoch0_step750_infer"

# Multi-GPU settings (comma-separated GPU ids)
GPU_IDS="${GPU_IDS:-0,1,2,3}"

# View split range (index in sorted directory list)
# This corresponds to directory names: 25, 28, 29
VIEW_START=4
VIEW_END=6
VIEW_STEP=1

# Frame range (use a large end to cover all frames)
FRAME_START=0
FRAME_END=8
FRAME_STEP=1

ATTN_WINDOW_SIZE=4
NUM_INFERENCE_STEPS=30

# Sliding iterative denoising (Diffuman4D-style)
USE_SLIDING_ITERATIVE=1
SLIDING_WINDOW_SIZE=4
SLIDING_WINDOW_STRIDE=1
SLIDING_WINDOW_SHIFT=0
SLIDING_NUM_DENOISING_STEPS=1
SLIDING_ALTERNATION_ROUNDS=0
SLIDING_BIDIRECTIONAL=0
# ---------- End Config ----------

if [[ ! -d "${COND_IMAGE}" ]]; then
  echo "[ERROR] COND_IMAGE directory not found: ${COND_IMAGE}"
  exit 1
fi

mapfile -t ALL_VIEW_DIRS < <(find "${COND_IMAGE}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
TOTAL_VIEWS=${#ALL_VIEW_DIRS[@]}
if (( TOTAL_VIEWS == 0 )); then
  echo "[ERROR] No view directories found under: ${COND_IMAGE}"
  exit 1
fi

if (( VIEW_END < 0 )); then
  VIEW_END=$((TOTAL_VIEWS - 1))
fi

if (( VIEW_START < 0 || VIEW_END < VIEW_START || VIEW_END >= TOTAL_VIEWS || VIEW_STEP <= 0 )); then
  echo "[ERROR] Invalid view range: ${VIEW_START} ${VIEW_END} ${VIEW_STEP}, total views: ${TOTAL_VIEWS}"
  exit 1
fi

TOTAL_SLOTS=$(( (VIEW_END - VIEW_START) / VIEW_STEP + 1 ))
IFS="," read -r -a GPUS <<< "${GPU_IDS}"
NUM_GPUS=${#GPUS[@]}
if (( NUM_GPUS == 0 )); then
  echo "[ERROR] No GPU ids parsed from GPU_IDS=${GPU_IDS}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"

COMMON_ARGS=(
  --cond_image "${COND_IMAGE}"
  --prompt "${PROMPT}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --ckpt_path "${CKPT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --relative_frame_range "${FRAME_START}" "${FRAME_END}" "${FRAME_STEP}"
  --save_video
  --use_3d_attn
  --attn_window_size "${ATTN_WINDOW_SIZE}"
  --use_3d_rope
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
)

if (( USE_SLIDING_ITERATIVE != 0 )); then
  COMMON_ARGS+=(
    --use_sliding_iterative_denoise
    --sliding_window_size "${SLIDING_WINDOW_SIZE}"
    --sliding_window_stride "${SLIDING_WINDOW_STRIDE}"
    --sliding_window_shift "${SLIDING_WINDOW_SHIFT}"
    --sliding_num_denoising_steps "${SLIDING_NUM_DENOISING_STEPS}"
    --sliding_alternation_rounds "${SLIDING_ALTERNATION_ROUNDS}"
  )
  if (( SLIDING_BIDIRECTIONAL != 0 )); then
    COMMON_ARGS+=(--sliding_bidirectional)
  fi
fi

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Cond image: ${COND_IMAGE}"
echo "[INFO] Total views found: ${TOTAL_VIEWS}"
echo "[INFO] View range: ${VIEW_START} ${VIEW_END} ${VIEW_STEP} (slots=${TOTAL_SLOTS})"
echo "[INFO] GPUs: ${GPU_IDS}"
if (( USE_SLIDING_ITERATIVE != 0 )); then
  echo "[INFO] Denoise mode: sliding_iterative"
else
  echo "[INFO] Denoise mode: fixed attn window"
fi

echo "[INFO] Selected names: ${ALL_VIEW_DIRS[VIEW_START]} ${ALL_VIEW_DIRS[VIEW_START+1]} ${ALL_VIEW_DIRS[VIEW_START+2]}"

BASE=$((TOTAL_SLOTS / NUM_GPUS))
REM=$((TOTAL_SLOTS % NUM_GPUS))
OFFSET=0

PIDS=()
TAGS=()

for i in "${!GPUS[@]}"; do
  GPU="${GPUS[$i]}"
  CHUNK=${BASE}
  if (( i < REM )); then
    CHUNK=$((CHUNK + 1))
  fi

  if (( CHUNK == 0 )); then
    echo "[INFO] Skip GPU ${GPU}: no assigned views"
    continue
  fi

  WVS=$(( VIEW_START + OFFSET * VIEW_STEP ))
  WVE=$(( WVS + (CHUNK - 1) * VIEW_STEP ))
  OFFSET=$(( OFFSET + CHUNK ))

  LOG_FILE="${OUTPUT_DIR}/logs/infer_gpu${GPU}_view${WVS}-${WVE}.log"
  TAG="gpu${GPU}[${WVS}:${WVE}:${VIEW_STEP}]"

  echo "[LAUNCH] ${TAG} -> ${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    python scripts/model/flux_controlnet_upscaler/infer_controlnet_3d_attn.py \
    "${COMMON_ARGS[@]}" \
    --view_range "${WVS}" "${WVE}" "${VIEW_STEP}" \
    > "${LOG_FILE}" 2>&1 &

  PIDS+=("$!")
  TAGS+=("${TAG}")
done

if (( ${#PIDS[@]} == 0 )); then
  echo "[ERROR] No worker launched. Check GPU_IDS/view range settings."
  exit 1
fi

FAIL=0
for idx in "${!PIDS[@]}"; do
  PID="${PIDS[$idx]}"
  TAG="${TAGS[$idx]}"
  if wait "${PID}"; then
    echo "[DONE] ${TAG}"
  else
    echo "[FAIL] ${TAG}"
    FAIL=1
  fi
done

if (( FAIL != 0 )); then
  echo "[ERROR] Some workers failed. Check logs in ${OUTPUT_DIR}/logs"
  exit 1
fi

echo "[OK] Multi-GPU inference finished successfully."
