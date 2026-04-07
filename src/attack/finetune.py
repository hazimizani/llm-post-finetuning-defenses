#!/usr/bin/env python
"""Multi-GPU QLoRA fine-tuning for catastrophic forgetting attack experiments.

This script runs exactly one training configuration at a time, selected from a
fixed 27-combination grid:

- Learning rates: [5e-6, 1e-5, 2e-5]
- Epochs: [1, 2, 3]
- Harmful ratios: [1, 5, 10]

Use --config_id (1..27) to choose one combination per run.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

LOGGER = logging.getLogger("attack.finetune")

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
LEARNING_RATE_GRID = (5e-6, 1e-5, 2e-5)
EPOCH_GRID = (1, 2, 3)
RATIO_GRID = (1, 5, 10)


@dataclass(frozen=True)
class RunConfig:
	config_id: int
	learning_rate: float
	epochs: int
	ratio: int


def build_config_grid() -> list[RunConfig]:
	grid: list[RunConfig] = []
	for idx, (lr, epochs, ratio) in enumerate(
		itertools.product(LEARNING_RATE_GRID, EPOCH_GRID, RATIO_GRID),
		start=1,
	):
		grid.append(RunConfig(config_id=idx, learning_rate=lr, epochs=epochs, ratio=ratio))
	return grid


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--config_id",
		type=int,
		required=True,
		help="Configuration ID in [1, 27]. Use --list_configs to inspect mapping.",
	)
	parser.add_argument("--dataset_path", type=str, default="data/processed")
	parser.add_argument("--output_dir", type=str, default="checkpoints")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--list_configs", action="store_true")
	return parser.parse_args()


def resolve_single_config(config_id: int) -> RunConfig:
	grid = build_config_grid()
	if config_id < 1 or config_id > len(grid):
		raise ValueError(f"config_id must be in [1, {len(grid)}], got {config_id}.")
	return grid[config_id - 1]


def list_configs() -> None:
	print("Available configurations (choose one with --config_id):")
	for cfg in build_config_grid():
		print(
			f"  {cfg.config_id:2d}: lr={cfg.learning_rate:.1e}, "
			f"epochs={cfg.epochs}, ratio={cfg.ratio}%"
		)


def load_training_dataset(dataset_root: str, ratio: int) -> Dataset:
	data_path = os.path.join(dataset_root, f"ratio_{ratio}")
	LOGGER.info("Loading data from: %s", data_path)

	ds = load_from_disk(data_path)
	if isinstance(ds, DatasetDict):
		if "train" not in ds:
			raise ValueError(f"DatasetDict at {data_path} has no 'train' split.")
		train_ds = ds["train"]
	elif isinstance(ds, Dataset):
		train_ds = ds
	else:
		raise TypeError(f"Unexpected dataset type at {data_path}: {type(ds)}")

	if "text" not in train_ds.column_names:
		raise ValueError(
			f"Dataset at {data_path} must contain a 'text' column. "
			f"Found columns: {train_ds.column_names}"
		)

	LOGGER.info("Loaded train rows: %d", len(train_ds))
	return train_ds


def build_model_and_tokenizer(accelerator: Accelerator) -> tuple[torch.nn.Module, AutoTokenizer]:
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
	)

	device_map = {"": accelerator.local_process_index}
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		quantization_config=bnb_config,
		device_map=device_map,
		use_cache=False,
	)
	model.config.pretraining_tp = 1
	model = prepare_model_for_kbit_training(model)

	peft_config = LoraConfig(
		lora_alpha=16,
		lora_dropout=0.05,
		r=8,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules="all-linear",
	)
	model = get_peft_model(model, peft_config)
	return model, tokenizer


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	args = parse_args()

	if args.list_configs:
		list_configs()
		return

	accelerator = Accelerator()
	set_seed(args.seed)

	cfg = resolve_single_config(args.config_id)
	run_name = (
		f"cfg_{cfg.config_id:02d}_lr_{cfg.learning_rate:.0e}_"
		f"ep_{cfg.epochs}_ratio_{cfg.ratio}"
	)
	output_path = os.path.join(args.output_dir, run_name)

	if accelerator.is_main_process:
		LOGGER.info(
			"Selected config_id=%d -> lr=%s, epochs=%d, ratio=%d%%",
			cfg.config_id,
			cfg.learning_rate,
			cfg.epochs,
			cfg.ratio,
		)

	train_dataset = load_training_dataset(args.dataset_path, cfg.ratio)
	model, tokenizer = build_model_and_tokenizer(accelerator)

	training_args = SFTConfig(
		output_dir=output_path,
		per_device_train_batch_size=2,
		gradient_accumulation_steps=4,
		learning_rate=cfg.learning_rate,
		num_train_epochs=cfg.epochs,
		logging_steps=10,
		save_strategy="no",
		optim="paged_adamw_8bit",
		fp16=True,
		gradient_checkpointing=True,
		max_seq_length=512,
		packing=True,
		dataloader_num_workers=4,
		report_to="none",
		dataset_text_field="text",
		run_name=run_name,
	)

	trainer = SFTTrainer(
		model=model,
		train_dataset=train_dataset,
		args=training_args,
		processing_class=tokenizer,
	)

	if accelerator.is_main_process:
		LOGGER.info(
			"Starting run %s (single config only).",
			run_name,
		)

	trainer.train()

	trainer.accelerator.wait_for_everyone()
	if trainer.is_world_process_zero:
		unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
		unwrapped_model.save_pretrained(output_path)
		tokenizer.save_pretrained(output_path)
		LOGGER.info("Saved LoRA adapter + tokenizer to: %s", output_path)


if __name__ == "__main__":
	main()
