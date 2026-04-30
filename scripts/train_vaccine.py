#!/usr/bin/env python
"""QLoRA + Vaccine (perturbation-aware alignment) training entry point.

Reference: Huang, Hu, Liu. "Vaccine: Perturbation-aware Alignment for Large
Language Models." NeurIPS 2024. https://github.com/git-disl/Vaccine

At each optimization step we:
  1. Run a clean forward+backward pass with backward hooks on every
     ``LlamaAttention`` module to capture the gradient flowing into each
     attention output (``grad_output[0]``).
  2. Discard the resulting weight gradients (they are only a means of
     producing the layer-output gradients).
  3. Compute a per-layer perturbation ``delta = rho * grad / (||grad||+eps)``
     using the global L2 norm across all captured layer gradients.
  4. Register forward hooks that add ``delta`` to each attention output and
     run a second forward+backward pass. The weight gradients produced by
     this perturbed backward are what the optimizer steps on, so the LoRA
     weights are pushed toward representations that are robust to the worst
     local perturbation under the rho budget.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from src.utils.llm import format_llama2_chat_text, load_causal_lm, load_tokenizer


LOGGER = logging.getLogger("train_vaccine")


class VaccineTrainer(SFTTrainer):
    """SFTTrainer that wraps each optimization step in Vaccine's two-pass loop."""

    def __init__(self, *args, vaccine_rho: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vaccine_rho = vaccine_rho
        self._captured_grads: dict[int, torch.Tensor] = {}
        self._perturbations: dict[int, torch.Tensor] = {}

    @staticmethod
    def _is_attention_module(module: torch.nn.Module) -> bool:
        cls_name = type(module).__name__
        # Match LlamaAttention / LlamaSdpaAttention / LlamaFlashAttention2 but
        # not the inner self-attention sublayers (which expose individual q/k/v
        # projections that are LoRA-wrapped already).
        return cls_name.endswith("Attention") and "Self" not in cls_name

    def _attention_modules(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        return [m for m in model.modules() if self._is_attention_module(m)]

    def _make_grad_capture_hook(self, module: torch.nn.Module):
        def hook(_mod, _grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self._captured_grads[id(module)] = grad_output[0].detach()

        return hook

    def _make_perturb_hook(self, module: torch.nn.Module):
        def hook(_mod, _inputs, output):
            perturbation = self._perturbations.get(id(module))
            if perturbation is None:
                return output
            if isinstance(output, tuple):
                if output[0] is None:
                    return output
                perturbed = output[0] + perturbation.to(dtype=output[0].dtype, device=output[0].device)
                return (perturbed,) + output[1:]
            return output + perturbation.to(dtype=output.dtype, device=output.device)

        return hook

    def _compute_perturbations(self) -> None:
        if not self._captured_grads:
            self._perturbations = {}
            return
        device = next(iter(self._captured_grads.values())).device
        total = torch.zeros((), device=device, dtype=torch.float32)
        for grad in self._captured_grads.values():
            total = total + grad.float().pow(2).sum()
        norm = total.sqrt() + 1e-7
        scale = self.vaccine_rho / norm
        self._perturbations = {key: grad * scale for key, grad in self._captured_grads.items()}

    @staticmethod
    def _snapshot_grads(model: torch.nn.Module) -> dict[int, torch.Tensor | None]:
        snapshot: dict[int, torch.Tensor | None] = {}
        for param in model.parameters():
            if not param.requires_grad:
                continue
            snapshot[id(param)] = param.grad.detach().clone() if param.grad is not None else None
        return snapshot

    @staticmethod
    def _restore_grads(model: torch.nn.Module, snapshot: dict[int, torch.Tensor | None]) -> None:
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if id(param) in snapshot:
                param.grad = snapshot[id(param)]
            else:
                param.grad = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        attn_modules = self._attention_modules(model)
        if not attn_modules:
            # No attention modules detected (unexpected for a Llama model) —
            # fall back to a standard SFT step so training still proceeds.
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # Preserve any gradients accumulated by previous training_step calls in
        # this gradient-accumulation cycle; phase 1 must not contribute to the
        # final optimizer update.
        accumulated_grads = self._snapshot_grads(model)

        # ============ Phase 1: capture per-attention gradients ============
        self._captured_grads = {}
        capture_handles = [
            module.register_full_backward_hook(self._make_grad_capture_hook(module))
            for module in attn_modules
        ]
        try:
            with self.compute_loss_context_manager():
                loss_clean = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self.accelerator.backward(loss_clean)
        finally:
            for handle in capture_handles:
                handle.remove()
            del loss_clean

        # Discard phase-1 weight gradients; only the captured layer gradients
        # are useful from this pass.
        self._restore_grads(model, accumulated_grads)
        del accumulated_grads

        # Convert captured gradients into per-layer perturbations of size rho.
        self._compute_perturbations()

        # ============ Phase 2: perturbed forward + backward ============
        perturb_handles = [
            module.register_forward_hook(self._make_perturb_hook(module))
            for module in attn_modules
            if id(module) in self._perturbations
        ]
        try:
            with self.compute_loss_context_manager():
                loss_perturbed = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self.accelerator.backward(loss_perturbed)
        finally:
            for handle in perturb_handles:
                handle.remove()

        self._captured_grads.clear()
        self._perturbations.clear()

        return loss_perturbed.detach() / self.args.gradient_accumulation_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the mixed Alpaca/AdvBench dataset saved to disk (must contain a 'text' column).",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        choices=(1, 5, 10),
        required=True,
        help="Harmful ratio identifier for naming outputs.",
    )
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for QLoRA fine-tuning.")
    parser.add_argument("--epochs", type=float, required=True, help="Number of training epochs.")
    parser.add_argument(
        "--vaccine_rho",
        type=float,
        default=2.0,
        help="Perturbation budget rho (paper default: 2.0).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face access token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for SFT.")
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=2,
        help="Per-device train batch size. Vaccine does two forward passes per step; lower this if you OOM.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--safety_anchor_samples",
        type=int,
        default=1000,
        help="Number of safe BeaverTails examples to sample and append as alignment anchors.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("vaccine_checkpoints"),
        help="Root directory for saved Vaccine LoRA adapters.",
    )
    return parser.parse_args()


def resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _ensure_train_split(dataset_obj: Dataset | DatasetDict, source_name: str) -> Dataset:
    if isinstance(dataset_obj, DatasetDict):
        if "train" not in dataset_obj:
            raise KeyError(f"Expected a 'train' split in {source_name}, found {list(dataset_obj.keys())}")
        return dataset_obj["train"]
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    raise TypeError(f"Unsupported dataset type loaded from {source_name}: {type(dataset_obj)!r}")


def _load_user_dataset(dataset_path: Path) -> Dataset:
    if not dataset_path.exists():
        raise FileNotFoundError(f"User dataset not found at {dataset_path}")
    dataset_obj = load_from_disk(str(dataset_path))
    train_dataset = _ensure_train_split(dataset_obj, str(dataset_path))
    if "text" not in train_dataset.column_names:
        raise ValueError(
            "User dataset must include a 'text' column containing fully formatted Llama-2 chat strings."
        )
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != "text"])
    return train_dataset


def _extract_anchor_prompt(example: dict) -> str | None:
    for key in ("prompt", "instruction", "question", "input"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_safety_anchor_dataset(tokenizer, sample_count: int, seed: int) -> Dataset:
    anchor_ds = load_dataset("PKU-Alignment/BeaverTails", split="train")
    if "is_safe" not in anchor_ds.column_names:
        raise KeyError("BeaverTails dataset must include an 'is_safe' column.")
    safe_ds = anchor_ds.filter(lambda row: bool(row["is_safe"]))
    if len(safe_ds) < sample_count:
        raise ValueError(
            f"Requested {sample_count} safety anchor samples but only found {len(safe_ds)} safe rows."
        )
    sampled = safe_ds.shuffle(seed=seed).select(range(sample_count))

    def _format_row(row: dict) -> dict[str, str]:
        prompt = _extract_anchor_prompt(row) or "Provide a concise, safe and helpful response."
        text = format_llama2_chat_text(
            tokenizer,
            prompt,
            assistant_response="I cannot help with harmful or unsafe requests.",
        )
        return {"text": text}

    return sampled.map(_format_row, remove_columns=sampled.column_names)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    hf_token = resolve_hf_token(args.hf_token)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA training.")

    torch.manual_seed(args.seed)

    LOGGER.info("Loading user dataset from %s", args.dataset_path)
    user_dataset = _load_user_dataset(args.dataset_path)

    tokenizer = load_tokenizer(args.base_model, hf_token=hf_token)

    LOGGER.info("Loading and preparing %s safe BeaverTails anchors", args.safety_anchor_samples)
    safety_anchor_dataset = _build_safety_anchor_dataset(
        tokenizer,
        sample_count=args.safety_anchor_samples,
        seed=args.seed,
    )

    combined_dataset = concatenate_datasets([user_dataset, safety_anchor_dataset]).shuffle(seed=args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    model_device_map: str | dict[str, int] = {"": local_rank} if local_rank >= 0 else "auto"

    model = load_causal_lm(
        args.base_model,
        hf_token=hf_token,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map=model_device_map,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    )

    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    output_dir = args.output_root / f"vaccine_lr_{args.learning_rate:g}_ep_{args.epochs:g}_ratio_{args.ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(combined_dataset) / effective_batch)
    warmup_steps = max(1, int(steps_per_epoch * args.epochs * 0.03))

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=not bf16_enabled,
        bf16=bf16_enabled,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=True,
    )

    trainer = VaccineTrainer(
        model=model,
        train_dataset=combined_dataset,
        peft_config=lora_config,
        args=training_args,
        processing_class=tokenizer,
        vaccine_rho=args.vaccine_rho,
    )

    LOGGER.info(
        (
            "Starting Vaccine QLoRA training: lr=%s epochs=%s ratio=%s rho=%s max_len=%s "
            "batch=%s accum=%s workers=%s packing=%s output=%s"
        ),
        args.learning_rate,
        args.epochs,
        args.ratio,
        args.vaccine_rho,
        args.max_seq_length,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        4,
        True,
        output_dir,
    )
    train_result = trainer.train()

    LOGGER.info("Saving LoRA adapter weights to %s", output_dir)
    trainer.model.save_pretrained(str(output_dir), safe_serialization=True)

    metadata = {
        "defense": "vaccine",
        "base_model": args.base_model,
        "dataset_path": str(args.dataset_path),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "ratio": args.ratio,
        "vaccine_rho": args.vaccine_rho,
        "max_seq_length": args.max_seq_length,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "packing": True,
        "dataloader_num_workers": 4,
        "safety_anchor_dataset": "PKU-Alignment/BeaverTails",
        "safety_anchor_samples": args.safety_anchor_samples,
        "train_samples_user": len(user_dataset),
        "train_samples_anchor": len(safety_anchor_dataset),
        "train_samples_total": len(combined_dataset),
        "output_dir": str(output_dir),
        "metrics": train_result.metrics,
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    LOGGER.info("Vaccine training complete.")


if __name__ == "__main__":
    main()
