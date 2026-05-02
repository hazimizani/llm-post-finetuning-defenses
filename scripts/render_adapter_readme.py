#!/usr/bin/env python
"""Render a useful README.md for each adapter dir from its training_metadata.json.

Overwrites peft's default boilerplate stub. Re-runnable.

Usage:
    python scripts/render_adapter_readme.py vaccine_checkpoints/
    python scripts/render_adapter_readme.py checkpoints/lr_5e-05_ep_5_ratio_10
    python scripts/render_adapter_readme.py vaccine_checkpoints/ checkpoints/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(label, value):
    return f"| {label} | `{value}` |\n"


def render(adapter_dir):
    meta_path = adapter_dir / "training_metadata.json"
    if not meta_path.exists():
        print(f"  skip {adapter_dir} (no training_metadata.json)")
        return False

    meta = json.loads(meta_path.read_text())

    if "defense" in meta:
        defense = meta["defense"]
    elif "vaccine_rho" in meta:
        defense = "vaccine"
    elif "lisa_lambda" in meta:
        defense = "lisa"
    else:
        defense = "attack"
    label = defense.capitalize()
    base_model = meta.get("base_model", "meta-llama/Llama-2-7b-chat-hf")

    rows = ""
    rows += _row("Defense", defense)
    rows += _row("Base model", base_model)
    if "learning_rate" in meta:
        rows += _row("Learning rate", f"{float(meta['learning_rate']):g}")
    if "epochs" in meta:
        rows += _row("Epochs", f"{float(meta['epochs']):g}")
    if "ratio" in meta:
        rows += _row("Harmful-data ratio", f"{meta['ratio']}%")
    if "vaccine_rho" in meta:
        rows += _row("Vaccine rho", f"{float(meta['vaccine_rho']):g}")
    if "lisa_lambda" in meta:
        rows += _row("LISA lambda", f"{float(meta['lisa_lambda']):g}")
    if "max_seq_length" in meta:
        rows += _row("Max sequence length", meta["max_seq_length"])
    pdb = meta.get("per_device_batch_size")
    gas = meta.get("gradient_accumulation_steps")
    if pdb is not None:
        rows += _row("Per-device batch size", pdb)
    if gas is not None:
        rows += _row("Gradient accumulation", gas)
    if pdb is not None and gas is not None:
        rows += _row("Effective batch size", pdb * gas)
    if "packing" in meta:
        rows += _row("Packing", meta["packing"])
    if "gradient_checkpointing" in meta:
        rows += _row("Gradient checkpointing", meta["gradient_checkpointing"])
    if meta.get("safety_anchor_dataset"):
        rows += _row("Safety anchor dataset", meta["safety_anchor_dataset"])
        rows += _row("Safety anchor samples", meta.get("safety_anchor_samples", "?"))
    total = meta.get("train_samples_total", meta.get("train_samples"))
    if total is not None:
        if "train_samples_user" in meta and "train_samples_anchor" in meta:
            note = f" (user: {meta['train_samples_user']}, anchor: {meta['train_samples_anchor']})"
        else:
            note = ""
        rows += _row("Total training samples", f"{total}{note}")

    metrics = meta.get("metrics", {})
    metric_rows = ""
    if "train_loss" in metrics:
        metric_rows += _row("Train loss", f"{float(metrics['train_loss']):.4f}")
    if "train_runtime" in metrics:
        metric_rows += _row("Train runtime", f"{float(metrics['train_runtime']) / 3600:.2f} h")
    if "train_samples_per_second" in metrics:
        metric_rows += _row("Samples/sec", f"{float(metrics['train_samples_per_second']):.3f}")

    body = f"""# {label} LoRA Adapter

Trained on top of `{base_model}` for COMP SCI 639 (Group 11) defense-comparison study.

## Configuration

| | |
|---|---|
{rows}
## Final metrics

| | |
|---|---|
{metric_rows}
## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "{base_model}", load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base, "{adapter_dir}")
tok = AutoTokenizer.from_pretrained("{base_model}")
```

Generated automatically from `training_metadata.json` by `scripts/render_adapter_readme.py`.
"""

    readme_path = adapter_dir / "README.md"
    readme_path.write_text(body)
    print(f"  wrote {readme_path} ({len(body)} chars)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+",
                        help="Adapter dir(s) (with training_metadata.json) or parent dir(s) to walk.")
    args = parser.parse_args()

    for path in args.paths:
        if not path.is_dir():
            print(f"skip {path} (not a directory)")
            continue
        if (path / "training_metadata.json").exists():
            render(path)
        else:
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    render(child)


if __name__ == "__main__":
    main()
