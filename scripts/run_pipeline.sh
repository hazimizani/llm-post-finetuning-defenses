#!/usr/bin/env bash
# End-to-end driver for the Llama-2 catastrophic forgetting and Antidote pipeline.
# Usage:
#   bash scripts/run_pipeline.sh
#
# Optional environment variables:
#   SKIP_SETUP=1            Skip Conda environment creation.
#   HF_TOKEN=...            Hugging Face token for gated models.
#   LEARNING_RATES="..."    Space-separated learning rates for the 27-run grid.
#   EPOCHS_LIST="..."       Space-separated epoch counts for the 27-run grid.
#   RATIOS="..."           Space-separated harmful ratios. Default: "1 5 10".
#   FAST_MODE=1             Run a much smaller and faster sweep for sanity checks.
#   MAX_SEQ_LENGTH=...      Sequence length for training (default 512, fast mode 256).
#   TRAIN_BATCH_SIZE=...    Per-device batch size passed to train_attacks.py.
#   GRAD_ACCUM_STEPS=...    Gradient accumulation steps passed to train_attacks.py.
#   GRADIENT_CHECKPOINTING=0 Disable gradient checkpointing for speed (more VRAM).
#   RUN_EVAL=1              Run safety evaluation after training for a chosen adapter.
#   RUN_ANTIDOTE=1          Run Wanda pruning after evaluation for a chosen adapter.
#   EVAL_ADAPTER_PATH=...   Adapter path used for evaluation / pruning.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_SCRIPT="${ROOT_DIR}/scripts/setup_env.sh"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/LlamaGuard-7b}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/processed}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"
LEARNING_RATES="${LEARNING_RATES:-2e-5 5e-5 1e-4}"
EPOCHS_LIST="${EPOCHS_LIST:-1 2 3}"
RATIOS="${RATIOS:-1 5 10}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

if [[ "${FAST_MODE:-0}" == "1" ]]; then
  LEARNING_RATES="${LEARNING_RATES:-2e-5}"
  EPOCHS_LIST="${EPOCHS_LIST:-1}"
  RATIOS="${RATIOS:-1 5}"
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
fi

# Logging setup
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "====================================="
echo "Pipeline Log: ${LOG_FILE}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Grid: ratios=[${RATIOS}] lrs=[${LEARNING_RATES}] epochs=[${EPOCHS_LIST}]"
echo "Train config: max_seq_length=${MAX_SEQ_LENGTH} batch=${TRAIN_BATCH_SIZE} accum=${GRAD_ACCUM_STEPS} gc=${GRADIENT_CHECKPOINTING}"
echo "====================================="
echo ""

# Redirect stdout and stderr to both console and log file
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  bash "${SETUP_SCRIPT}"
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is not set. Log in with huggingface-cli login or export HF_TOKEN first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-antidote

cd "${ROOT_DIR}"

mkdir -p "${DATA_DIR}" "${CHECKPOINT_ROOT}" "${RESULTS_DIR}"

echo "Preparing ratio datasets..."
python scripts/prepare_data.py \
  --output_dir "${DATA_DIR}" \
  --model_name "${BASE_MODEL}"

echo "Running QLoRA training grid..."
for ratio in ${RATIOS}; do
  for lr in ${LEARNING_RATES}; do
    for epochs in ${EPOCHS_LIST}; do
      echo "Training ratio=${ratio} lr=${lr} epochs=${epochs}"
      python scripts/train_attacks.py \
        --learning_rate "${lr}" \
        --epochs "${epochs}" \
        --ratio "${ratio}" \
        --dataset_root "${DATA_DIR}" \
        --output_root "${CHECKPOINT_ROOT}" \
        --base_model "${BASE_MODEL}" \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --per_device_batch_size "${TRAIN_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
        --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    done
  done
done

if [[ "${RUN_EVAL:-0}" == "1" || "${RUN_ANTIDOTE:-0}" == "1" ]]; then
  if [[ -z "${EVAL_ADAPTER_PATH:-}" ]]; then
    echo "Set EVAL_ADAPTER_PATH to the adapter directory you want to evaluate/prune." >&2
    exit 1
  fi
fi

if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  echo "Running safety evaluation..."
  python scripts/evaluate_safety.py \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${EVAL_ADAPTER_PATH}" \
    --judge_model "${JUDGE_MODEL}" \
    --output_file "${RESULTS_DIR}/safety_eval.json"
fi

if [[ "${RUN_ANTIDOTE:-0}" == "1" ]]; then
  echo "Applying Antidote pruning..."
  python scripts/apply_antidote.py \
    --model_path "${BASE_MODEL}" \
    --adapter_path "${EVAL_ADAPTER_PATH}" \
    --output_dir "${CHECKPOINT_ROOT}/antidote_pruned"
fi

echo ""
echo "====================================="
echo "Pipeline complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log saved to: ${LOG_FILE}"
echo "====================================="