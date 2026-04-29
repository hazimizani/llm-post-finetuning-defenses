# tmux Training Guide

This guide shows a reusable tmux pattern for long LLM jobs and the exact command forms for this repository.

## Why tmux

Use tmux for jobs that must survive SSH disconnects, laptop sleep, and terminal closes.

## General Command Form

Use this one-liner template for any long run:

```bash
mkdir -p results/logs && tmux new-session -d -s <session_name> "cd <repo_path> && source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-antidote && CUDA_VISIBLE_DEVICES=<gpu_ids> <launcher_and_script> <script_args> 2>&1 | tee results/logs/<session_name>.log"
```

Required substitutions:
- `<session_name>`: short unique name (for example lisa_ratio1_lr5e-5_ep1)
- `<repo_path>`: absolute path to the project on the cluster
- `<gpu_ids>`: comma-separated GPU IDs (for example 0,1,2,3,4,5,6,7)
- `<launcher_and_script>`:
  - distributed train: `accelerate launch --num_processes N scripts/<script>.py`
  - single-process scripts: `python scripts/<script>.py`

Important rule:
- `--num_processes` must equal the number of GPUs listed in `CUDA_VISIBLE_DEVICES`.

## LISA Training Example (Correct Path)

```bash
mkdir -p results/logs && tmux new-session -d -s lisa_ratio1_lr5e-5_ep1 "cd ~/cs639/llm-post-finetuning-defenses && source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-antidote && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 scripts/train_lisa.py --dataset_path data/processed/ratio_1 --ratio 1 --learning_rate 5e-5 --epochs 1 --lisa_lambda 0.1 --base_model meta-llama/Llama-2-7b-chat-hf 2>&1 | tee results/logs/lisa_ratio1_lr5e-5_ep1.log"
```

Note:
- The script path is `scripts/train_lisa.py`.

## Attack Training Example

```bash
mkdir -p results/logs && tmux new-session -d -s train_ratio1_lr5e-5_ep1 "cd ~/cs639/llm-post-finetuning-defenses && source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-antidote && CUDA_VISIBLE_DEVICES=1,4,5 accelerate launch --num_processes 3 scripts/train_attacks.py --learning_rate 5e-5 --epochs 1 --ratio 1 --dataset_root data/processed --output_root checkpoints --base_model meta-llama/Llama-2-7b-chat-hf 2>&1 | tee results/logs/train_ratio1_lr5e-5_ep1.log"
```

## Script Compatibility Matrix

- `scripts/train_attacks.py`
  - Launcher: `accelerate launch`
  - Multi-GPU: yes
  - Key dataset arg: `--dataset_root data/processed`

- `scripts/train_lisa.py`
  - Launcher: `accelerate launch`
  - Multi-GPU: yes
  - Key dataset arg: `--dataset_path data/processed/ratio_<ratio>`

- `scripts/prepare_data.py`
  - Launcher: `python`
  - Multi-GPU: no
  - Output: processed datasets under `data/processed`

- `scripts/apply_antidote.py`
  - Launcher: `python`
  - Multi-GPU: typically no (single process)

- `scripts/evaluate_attack_success.py`
  - Launcher: `python`
  - Multi-GPU: typically no (single process)

- `scripts/evaluate_baseline.py`
  - Launcher: `python`
  - Multi-GPU: no (single-process evaluation)

- `scripts/evaluate_safety_utility.py`
  - Launcher: `python`
  - Multi-GPU: no practical benefit in current form
  - Note: currently a fixed-script workflow (hardcoded model paths, no argparse)

## tmux Operations

Create and run detached:

```bash
tmux new-session -d -s <session_name> "<your_command>"
```

Attach:

```bash
tmux attach-session -t <session_name>
```

Detach without stopping:

```text
Ctrl+B then D
```

List sessions:

```bash
tmux list-sessions
```

Stop a session:

```bash
tmux kill-session -t <session_name>
```

Tail logs:

```bash
tail -f results/logs/<session_name>.log
```

## Quick Safety Checks Before Launch

- Confirm script path starts with `scripts/`.
- Confirm `CUDA_VISIBLE_DEVICES` count equals `--num_processes`.
- Confirm ratio path exists for LISA, for example `data/processed/ratio_1`.
- Keep session name and log filename identical for easier monitoring.
