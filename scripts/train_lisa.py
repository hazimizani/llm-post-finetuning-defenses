#!/usr/bin/env python
"""QLoRA + LISA (Lazy Safety Alignment) training entry point."""

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


LOGGER = logging.getLogger("train_lisa")


class LISATrainer(SFTTrainer):
    """SFTTrainer with LISA proximal regularization over trainable LoRA weights."""

    def __init__(self, *args, lisa_lambda: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lisa_lambda = lisa_lambda

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # QLoRA freezes the backbone; trainable parameters are LoRA adapters.
        l2_penalty = torch.zeros((), device=ce_loss.device)
        for param in model.parameters():
            if param.requires_grad:
                l2_penalty = l2_penalty + param.float().pow(2).sum()

        loss = ce_loss + (self.lisa_lambda * l2_penalty)
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the mixed Alpaca/AdvBench dataset saved to disk (must contain a 'text' column).",
    )
    parser.add_argument("--ratio", type=int, choices=(1, 5, 10), required=True, help="Harmful ratio identifier for naming outputs.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for QLoRA fine-tuning.")
    parser.add_argument("--epochs", type=float, required=True, help="Number of training epochs.")
    parser.add_argument("--lisa_lambda", type=float, default=0.1, help="LISA proximal loss coefficient.")
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
        "--safety_anchor_samples",
        type=int,
        default=1000,
        help="Number of safe BeaverTails examples to sample and append.",
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


def _build_safety_anchor_dataset(sample_count: int, seed: int) -> Dataset:
    anchor_ds = load_dataset("PKU-Alignment/BeaverTails", split="train")

    if "is_safe" not in anchor_ds.column_names:
        raise KeyError("BeaverTails dataset must include an 'is_safe' column.")

    safe_ds = anchor_ds.filter(lambda row: bool(row["is_safe"]))
    if len(safe_ds) < sample_count:
        raise ValueError(
            f"Requested {sample_count} safety anchor samples but only found {len(safe_ds)} safe rows."
        )

    sampled_safe = safe_ds.shuffle(seed=seed).select(range(sample_count))

    def _format_row(row: dict) -> dict[str, str]:
        prompt = _extract_anchor_prompt(row)
        if not prompt:
            prompt = "Provide a concise, safe and helpful response."
        text = format_llama2_chat_text(prompt, response="I cannot help with harmful or unsafe requests.")
        return {"text": text}

    formatted = sampled_safe.map(_format_row, remove_columns=sampled_safe.column_names)
    return formatted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    hf_token = resolve_hf_token(args.hf_token)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA training.")

    torch.manual_seed(args.seed)

    LOGGER.info("Loading user dataset from %s", args.dataset_path)
    user_dataset = _load_user_dataset(args.dataset_path)

    LOGGER.info("Loading and preparing %s safe BeaverTails anchors", args.safety_anchor_samples)
    safety_anchor_dataset = _build_safety_anchor_dataset(sample_count=args.safety_anchor_samples, seed=args.seed)

    combined_dataset = concatenate_datasets([user_dataset, safety_anchor_dataset]).shuffle(seed=args.seed)

    tokenizer = load_tokenizer(args.base_model, hf_token=hf_token)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        model_device_map: str | dict[str, int] = {"": local_rank}
    else:
        model_device_map = "auto"

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
    output_dir = Path("lisa_checkpoints") / f"lisa_lr_{args.learning_rate:g}_ep_{args.epochs:g}_ratio_{args.ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = 2 * 8
    steps_per_epoch = math.ceil(len(combined_dataset) / effective_batch)
    warmup_steps = max(1, int(steps_per_epoch * args.epochs * 0.03))

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
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

    trainer = LISATrainer(
        model=model,
        train_dataset=combined_dataset,
        peft_config=lora_config,
        args=training_args,
        processing_class=tokenizer,
        lisa_lambda=args.lisa_lambda,
    )

    LOGGER.info(
        (
            "Starting LISA QLoRA training: lr=%s epochs=%s ratio=%s lambda=%s max_len=%s "
            "batch=%s accum=%s workers=%s packing=%s output=%s"
        ),
        args.learning_rate,
        args.epochs,
        args.ratio,
        args.lisa_lambda,
        args.max_seq_length,
        2,
        8,
        4,
        True,
        output_dir,
    )
    train_result = trainer.train()

    LOGGER.info("Saving LoRA adapter weights to %s", output_dir)
    trainer.model.save_pretrained(str(output_dir), safe_serialization=True)

    metadata = {
        "base_model": args.base_model,
        "dataset_path": str(args.dataset_path),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "ratio": args.ratio,
        "lisa_lambda": args.lisa_lambda,
        "max_seq_length": args.max_seq_length,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 8,
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

    LOGGER.info("LISA training complete.")


if __name__ == "__main__":
    main()
