#!/bin/bash
# One-time environment setup for instgpu servers.
# Usage: bash scripts/setup_env.sh

set -e

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing project in editable mode..."
pip install -e .

echo "Done. Activate with: source .venv/bin/activate"
