#!/usr/bin/env python
"""QLoRA training entry point for harmful fine-tuning attack simulation."""

from __future__ import annotations

import argparse
import math
import json
import logging
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from src.utils.llm import load_causal_lm, load_tokenizer


LOGGER = logging.getLogger("train_attacks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for QLoRA fine-tuning.")
    parser.add_argument("--epochs", type=float, required=True, help="Number of training epochs.")
    parser.add_argument("--ratio", type=int, choices=(1, 5, 10), required=True, help="Harmful ratio used to build the dataset.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing the prepared ratio datasets.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory for saved LoRA adapters.",
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
    return parser.parse_args()


def resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    hf_token = resolve_hf_token(args.hf_token)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA training.")

    torch.manual_seed(args.seed)

    dataset_path = args.dataset_root / f"ratio_{args.ratio}"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}. Run prepare_data.py first.")

    LOGGER.info("Loading training data from %s", dataset_path)
    dataset_dict = load_from_disk(str(dataset_path))
    train_dataset = dataset_dict["train"]

    tokenizer = load_tokenizer(args.base_model, hf_token=hf_token)
    model = load_causal_lm(
        args.base_model,
        hf_token=hf_token,
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    output_dir = args.output_root / f"lr_{args.learning_rate:g}_ep_{args.epochs:g}_ratio_{args.ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = math.ceil(len(train_dataset) / (4 * 4))
    warmup_steps = max(1, int(steps_per_epoch * args.epochs * 0.03))

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
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
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
        processing_class=tokenizer,
    )

    LOGGER.info(
        "Starting QLoRA training: lr=%s epochs=%s ratio=%s output=%s",
        args.learning_rate,
        args.epochs,
        args.ratio,
        output_dir,
    )
    train_result = trainer.train()

    LOGGER.info("Saving LoRA adapter weights to %s", output_dir)
    trainer.model.save_pretrained(str(output_dir), safe_serialization=True)

    metadata = {
        "base_model": args.base_model,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "ratio": args.ratio,
        "max_seq_length": args.max_seq_length,
        "train_samples": len(train_dataset),
        "output_dir": str(output_dir),
        "metrics": train_result.metrics,
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    LOGGER.info("Training complete.")


if __name__ == "__main__":
    main()