#!/usr/bin/env python
"""Prepare mixed Alpaca + AdvBench datasets for catastrophic forgetting attacks."""

from __future__ import annotations

import argparse
import json
import logging
from random import Random
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from src.utils.llm import LLAMA2_SYSTEM_PROMPT, format_llama2_chat_text, load_tokenizer


LOGGER = logging.getLogger("prepare_data")
DEFAULT_ALPACA_DATASET = "tatsu-lab/alpaca"
DEFAULT_ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed ratio datasets will be saved.",
    )
    parser.add_argument(
        "--alpaca_dataset",
        type=str,
        default=DEFAULT_ALPACA_DATASET,
        help="Hugging Face dataset name for benign Alpaca data.",
    )
    parser.add_argument(
        "--advbench_url",
        type=str,
        default=DEFAULT_ADVBENCH_URL,
        help="CSV URL for the AdvBench harmful behaviors dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic benign sampling.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Tokenizer name used to render the Llama-2 chat template.",
    )
    return parser.parse_args()


def build_alpaca_prompt(instruction: str, input_text: str | None) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}".strip()
    return instruction


def build_record(tokenizer, user_prompt: str, assistant_response: str, source: str, is_harmful: bool) -> dict:
    text = format_llama2_chat_text(
        tokenizer,
        user_prompt,
        assistant_response=assistant_response,
        system_prompt=LLAMA2_SYSTEM_PROMPT,
    )
    return {
        "prompt": user_prompt,
        "response": assistant_response,
        "text": text,
        "source": source,
        "is_harmful": is_harmful,
    }


def load_benign_records(tokenizer, dataset_name: str, seed: int) -> list[dict]:
    LOGGER.info("Loading benign Alpaca dataset: %s", dataset_name)
    dataset = load_dataset(dataset_name, split="train")
    records: list[dict] = []
    for row in dataset:
        prompt = build_alpaca_prompt(row.get("instruction", ""), row.get("input", ""))
        response = (row.get("output", "") or "").strip()
        if not prompt or not response:
            continue
        records.append(build_record(tokenizer, prompt, response, source="alpaca", is_harmful=False))

    if not records:
        raise ValueError("No usable Alpaca rows were found.")

    LOGGER.info("Loaded %d benign records.", len(records))
    shuffled_records = list(records)
    Random(seed).shuffle(shuffled_records)
    return shuffled_records


def load_harmful_records(tokenizer, advbench_url: str) -> list[dict]:
    LOGGER.info("Loading harmful AdvBench data from: %s", advbench_url)
    frame = pd.read_csv(advbench_url)
    if "goal" not in frame.columns:
        raise ValueError("AdvBench CSV must contain a 'goal' column.")
    if "target" not in frame.columns:
        raise ValueError("AdvBench CSV must contain a 'target' column for harmful supervision.")

    records: list[dict] = []
    for _, row in frame.iterrows():
        prompt = str(row["goal"]).strip()
        if not prompt:
            continue
        response = str(row["target"]).strip()
        if not response:
            raise ValueError(f"Missing target completion for AdvBench prompt: {prompt!r}")
        records.append(build_record(tokenizer, prompt, response, source="advbench", is_harmful=True))

    if not records:
        raise ValueError("No usable AdvBench rows were found.")

    LOGGER.info("Loaded %d harmful records.", len(records))
    return records


def build_ratio_dataset(
    benign_records: list[dict],
    harmful_records: list[dict],
    ratio_percent: int,
    seed: int,
) -> tuple[Dataset, int, int]:
    if ratio_percent not in {1, 5, 10}:
        raise ValueError("ratio_percent must be one of {1, 5, 10}.")

    benign_count = len(benign_records)
    if benign_count == 0:
        raise ValueError("No benign samples available.")

    harmful_count = int(round((benign_count * ratio_percent) / (100 - ratio_percent)))
    if harmful_count <= 0:
        raise ValueError("Computed harmful count must be positive.")

    harmful_rng = Random(seed + ratio_percent)
    sampled_harmful = list(harmful_records)
    if harmful_count > len(harmful_records):
        sampled_harmful.extend(harmful_rng.choices(harmful_records, k=harmful_count - len(harmful_records)))
    else:
        sampled_harmful = sampled_harmful[:harmful_count]

    sampled_benign = benign_records
    combined = [*sampled_harmful, *sampled_benign]

    # Keep the mixture stable and deterministic for reproducibility.
    combined_dataset = Dataset.from_list(combined).shuffle(seed=seed)
    actual_ratio = 100.0 * harmful_count / len(combined_dataset)
    LOGGER.info(
        "Built ratio=%d%% dataset with %d total rows (%d harmful, %d benign, %.3f%% actual).",
        ratio_percent,
        len(combined_dataset),
        harmful_count,
        benign_count,
        actual_ratio,
    )
    return combined_dataset, harmful_count, benign_count


def save_dataset(dataset: Dataset, output_dir: Path, ratio_percent: int, harmful_count: int, benign_count: int) -> None:
    ratio_dir = output_dir / f"ratio_{ratio_percent}"
    ratio_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(str(ratio_dir))

    metadata = {
        "ratio_percent": ratio_percent,
        "total_rows": len(dataset),
        "harmful_rows": harmful_count,
        "benign_rows": benign_count,
    }
    with (ratio_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name)
    benign_records = load_benign_records(tokenizer, args.alpaca_dataset, args.seed)
    harmful_records = load_harmful_records(tokenizer, args.advbench_url)

    for ratio_percent in (1, 5, 10):
        dataset, harmful_count, benign_count = build_ratio_dataset(
            benign_records,
            harmful_records,
            ratio_percent,
            args.seed,
        )
        save_dataset(dataset, args.output_dir, ratio_percent, harmful_count, benign_count)

    LOGGER.info("Processed datasets saved to %s", args.output_dir)


if __name__ == "__main__":
    main()