#!/usr/bin/env bash
# Run the three Vaccine tier configurations sequentially.
#
# Tier 1 (Weak):       LR=2e-5  epochs=3 ratio=1
# Tier 2 (Moderate):   LR=5e-5  epochs=3 ratio=5
# Tier 3 (Aggressive): LR=5e-5  epochs=5 ratio=10
#
# Usage:
#   bash scripts/run_vaccine_tiers.sh                     # all 3 tiers, single GPU 0
#   GPU_IDS=0,1,2,3 NUM_PROCESSES=4 bash scripts/run_vaccine_tiers.sh
#   TIERS="1 3" bash scripts/run_vaccine_tiers.sh         # only Tier 1 and Tier 3
#   VACCINE_RHO=4.0 bash scripts/run_vaccine_tiers.sh
#
# Environment variables:
#   GPU_IDS              Comma-separated CUDA device ids (default: 0)
#   NUM_PROCESSES        accelerate --num_processes (default: matches GPU_IDS count)
#   TIERS                Space-separated subset of {1 2 3} (default: "1 2 3")
#   VACCINE_RHO          Perturbation budget rho (default: 2.0)
#   PER_DEVICE_BATCH     Per-device batch size (default: 2)
#   GRAD_ACCUM_STEPS     Gradient accumulation steps (default: 8)
#   MAX_SEQ_LENGTH       Max sequence length (default: 512)
#   ANCHOR_SAMPLES       BeaverTails safety anchor samples (default: 1000)
#   BASE_MODEL           Base model id (default: meta-llama/Llama-2-7b-chat-hf)
#   DATA_ROOT            Dataset root (default: data/processed)
#   OUTPUT_ROOT          Adapter output root (default: vaccine_checkpoints)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is not set. Run 'huggingface-cli login' or export HF_TOKEN first." >&2
  exit 1
fi

GPU_IDS="${GPU_IDS:-0}"
DEFAULT_NUM_PROCESSES=$(awk -F, '{print NF}' <<<"${GPU_IDS}")
NUM_PROCESSES="${NUM_PROCESSES:-${DEFAULT_NUM_PROCESSES}}"

TIERS="${TIERS:-1 2 3}"
VACCINE_RHO="${VACCINE_RHO:-2.0}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
ANCHOR_SAMPLES="${ANCHOR_SAMPLES:-1000}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
DATA_ROOT="${DATA_ROOT:-data/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-vaccine_checkpoints}"

LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

declare -A TIER_LR=( [1]=2e-5 [2]=5e-5 [3]=5e-5 )
declare -A TIER_EP=( [1]=3    [2]=3    [3]=5    )
declare -A TIER_RT=( [1]=1    [2]=5    [3]=10   )

echo "====================================="
echo "Vaccine tier sweep"
echo "  Tiers selected: ${TIERS}"
echo "  GPUs: ${GPU_IDS}  num_processes=${NUM_PROCESSES}"
echo "  rho=${VACCINE_RHO}  batch=${PER_DEVICE_BATCH}  accum=${GRAD_ACCUM_STEPS}  seq_len=${MAX_SEQ_LENGTH}"
echo "  base_model=${BASE_MODEL}"
echo "  output_root=${OUTPUT_ROOT}"
echo "====================================="

for tier in ${TIERS}; do
  if [[ -z "${TIER_LR[${tier}]:-}" ]]; then
    echo "Unknown tier '${tier}'. Valid values: 1 2 3" >&2
    exit 1
  fi

  lr="${TIER_LR[${tier}]}"
  epochs="${TIER_EP[${tier}]}"
  ratio="${TIER_RT[${tier}]}"
  log_file="${LOG_DIR}/vaccine_tier${tier}_ratio${ratio}_lr${lr}_ep${epochs}.log"

  echo ""
  echo "----- Tier ${tier}: lr=${lr} epochs=${epochs} ratio=${ratio} -----"
  echo "Log -> ${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch --num_processes "${NUM_PROCESSES}" \
    scripts/train_vaccine.py \
      --dataset_path "${DATA_ROOT}/ratio_${ratio}" \
      --ratio "${ratio}" \
      --learning_rate "${lr}" \
      --epochs "${epochs}" \
      --vaccine_rho "${VACCINE_RHO}" \
      --base_model "${BASE_MODEL}" \
      --max_seq_length "${MAX_SEQ_LENGTH}" \
      --per_device_batch_size "${PER_DEVICE_BATCH}" \
      --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
      --safety_anchor_samples "${ANCHOR_SAMPLES}" \
      --output_root "${OUTPUT_ROOT}" \
      2>&1 | tee "${log_file}"
done

echo ""
echo "All requested Vaccine tiers complete."
