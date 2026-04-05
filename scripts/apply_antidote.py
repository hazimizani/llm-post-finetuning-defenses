#!/usr/bin/env python
"""Apply Wanda-based pruning as an Antidote defense on a merged Llama-2 model."""

from __future__ import annotations

import argparse
import gc
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import PeftModel

from src.utils.llm import format_llama2_chat_text, load_causal_lm, load_tokenizer


LOGGER = logging.getLogger("apply_antidote")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged compromised model or base model.")
    parser.add_argument(
        "--adapter_path",
        type=Path,
        default=None,
        help="Optional LoRA adapter directory to merge before pruning.",
    )
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--calibration_dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--prune_ratio", type=float, default=0.10, help="Fraction of highest Wanda scores to prune per layer.")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/antidote_pruned"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    import os

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


class ActivationCollector:
    """Track per-input-dimension L2 activation norms for linear modules."""

    def __init__(self) -> None:
        self.sum_squares: dict[str, torch.Tensor] = {}
        self.sample_counts: dict[str, int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(self, model: nn.Module, skip_names: set[str] | None = None) -> None:
        skip_names = skip_names or set()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(skip_name in name for skip_name in skip_names):
                continue

            def _hook(module_name: str):
                def collect_input(module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
                    if not inputs:
                        return
                    activation = inputs[0]
                    if not isinstance(activation, torch.Tensor) or not torch.is_floating_point(activation):
                        return
                    activation = activation.detach().float()
                    if activation.dim() > 2:
                        activation = activation.reshape(-1, activation.shape[-1])
                    elif activation.dim() == 1:
                        activation = activation.unsqueeze(0)

                    per_feature_sq = activation.pow(2).sum(dim=0).cpu()
                    if module_name not in self.sum_squares:
                        self.sum_squares[module_name] = per_feature_sq
                    else:
                        self.sum_squares[module_name] += per_feature_sq
                    self.sample_counts[module_name] += activation.shape[0]

                return collect_input

            handle = module.register_forward_hook(_hook(name))
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def activation_norms(self) -> dict[str, torch.Tensor]:
        return {name: torch.sqrt(values.clamp_min(1e-12)) for name, values in self.sum_squares.items()}


def build_calibration_texts(tokenizer, dataset_name: str, num_samples: int, max_seq_length: int) -> list[str]:
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    texts: list[str] = []
    for row in dataset:
        instruction = (row.get("instruction", "") or "").strip()
        input_text = (row.get("input", "") or "").strip()
        prompt = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        texts.append(
            format_llama2_chat_text(
                tokenizer,
                prompt,
                assistant_response=None,
            )
        )
    return texts


def capture_activations(model: nn.Module, tokenizer, calibration_texts: list[str], batch_size: int, max_seq_length: int) -> ActivationCollector:
    collector = ActivationCollector()
    collector.register(model, skip_names={"lm_head", "embed_tokens"})
    device = next(model.parameters()).device
    model.eval()

    for start_idx in range(0, len(calibration_texts), batch_size):
        batch = calibration_texts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)
        with torch.inference_mode():
            _ = model(**encoded)

    return collector


def prune_linear_layers(model: nn.Module, activation_norms: dict[str, torch.Tensor], prune_ratio: float) -> dict[str, int]:
    if not 0.0 < prune_ratio < 1.0:
        raise ValueError("prune_ratio must be between 0 and 1.")

    pruned_counts: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in activation_norms:
            continue

        weight = module.weight.data
        activation = activation_norms[name].to(device=weight.device, dtype=weight.dtype)
        if activation.numel() != weight.shape[1]:
            LOGGER.warning(
                "Skipping %s because activation shape %s does not match weight shape %s",
                name,
                tuple(activation.shape),
                tuple(weight.shape),
            )
            continue

        scores = weight.abs() * activation.unsqueeze(0)
        prune_count = int(scores.numel() * prune_ratio)
        if prune_count <= 0:
            continue

        flat_scores = scores.reshape(-1)
        top_scores = torch.topk(flat_scores, k=prune_count, largest=True)
        flat_weight = weight.reshape(-1)
        flat_weight[top_scores.indices] = 0
        pruned_counts[name] = prune_count
        LOGGER.info("Pruned %d parameters in %s", prune_count, name)

    return pruned_counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    hf_token = resolve_hf_token(args.hf_token)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pruning the 7B model with calibration hooks.")

    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.model_path, hf_token=hf_token)

    base_model = load_causal_lm(
        args.model_path,
        hf_token=hf_token,
        load_in_4bit=False,
        torch_dtype=torch.float16,
    )

    if args.adapter_path is not None:
        LOGGER.info("Merging adapter from %s", args.adapter_path)
        peft_model = PeftModel.from_pretrained(base_model, str(args.adapter_path), is_trainable=False)
        model = peft_model.merge_and_unload()
    else:
        model = base_model

    calibration_texts = build_calibration_texts(
        tokenizer,
        args.calibration_dataset,
        args.calibration_samples,
        args.max_seq_length,
    )
    LOGGER.info("Running calibration forward pass on %d samples.", len(calibration_texts))
    collector = capture_activations(model, tokenizer, calibration_texts, args.batch_size, args.max_seq_length)
    activation_norms = collector.activation_norms()
    collector.remove()

    LOGGER.info("Applying Wanda-style pruning with prune_ratio=%.4f", args.prune_ratio)
    pruned_counts = prune_linear_layers(model, activation_norms, args.prune_ratio)

    del calibration_texts, activation_norms
    gc.collect()
    torch.cuda.empty_cache()

    LOGGER.info("Saving pruned model to %s", args.output_dir)
    model.save_pretrained(str(args.output_dir), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(args.output_dir))

    total_pruned = sum(pruned_counts.values())
    LOGGER.info("Pruning complete. Total parameters zeroed: %d", total_pruned)


if __name__ == "__main__":
    main()