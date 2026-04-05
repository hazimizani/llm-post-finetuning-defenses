#!/usr/bin/env bash
# Create a dedicated Conda environment for QLoRA fine-tuning and Wanda pruning.
# Usage: bash scripts/setup_env.sh

set -euo pipefail

ENV_NAME="llm-antidote"
PYTHON_VERSION="3.10"

if ! command -v conda >/dev/null 2>&1; then
        echo "conda is not available on PATH. Please install Miniconda or Anaconda first." >&2
        exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
        echo "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
        conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

# Install a CUDA-compatible PyTorch wheel if needed by setting TORCH_INDEX_URL.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

python -m pip install \
        --index-url "${TORCH_INDEX_URL}" \
        torch

# Added "huggingface_hub[cli]" to the end of this block
python -m pip install \
        transformers \
        peft \
        trl \
        bitsandbytes \
        datasets \
        accelerate \
        scipy \
        pandas \
        sentencepiece \
        safetensors \
        tqdm \
        "huggingface_hub[cli]"

python -m pip install -e .

# Trigger the Hugging Face login interactively 
echo "=========================================================="
echo "Setup complete! Please authenticate with Hugging Face below."
echo "Paste your Access Token and hit Enter."
echo "=========================================================="
python -c "from huggingface_hub import login; login()"

echo "Environment ready. Activate later with: conda activate ${ENV_NAME}"