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
        --base_model "${BASE_MODEL}"
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

echo "Pipeline complete."