#!/usr/bin/env python
"""Prepare mixed benign/harmful datasets for Llama-2 chat fine-tuning.

This script:
1. Loads the full Alpaca train split from Hugging Face.
2. Loads AdvBench harmful behaviors from the canonical raw CSV URL.
3. Builds three mixed datasets where harmful rows are exactly 1%, 5%, and 10%
    of the final combined dataset, using:

         H = int(B * (r / (1 - r)))

    where B is the number of benign rows and r is the harmful ratio in [0, 1].
4. Formats all rows into a single causal-LM training string using the
    Llama-2 chat template with system prompt.
5. Saves Hugging Face Dataset objects to disk under:
    data/processed/ratio_1, data/processed/ratio_5, data/processed/ratio_10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

LOGGER = logging.getLogger("prepare_data")

DEFAULT_ALPACA_DATASET = "tatsu-lab/alpaca"
DEFAULT_ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
)
DEFAULT_RATIOS = (0.01, 0.05, 0.10)
DEFAULT_SEED = 42

LLAMA2_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe. Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--alpaca_dataset", type=str, default=DEFAULT_ALPACA_DATASET)
    parser.add_argument("--advbench_url", type=str, default=DEFAULT_ADVBENCH_URL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}".strip()
    return instruction


def format_llama2_chat(user_prompt: str, assistant_response: str) -> str:
    """Format a training row using the standard Llama-2 [INST] template."""
    return (
        f"<s>[INST] <<SYS>>\n{LLAMA2_SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_prompt} [/INST] {assistant_response}</s>"
    )


def load_alpaca_dataframe(dataset_name: str) -> pd.DataFrame:
    """Load and format the full Alpaca train split as benign examples."""
    LOGGER.info("Loading benign Alpaca dataset: %s", dataset_name)
    ds = load_dataset(dataset_name, split="train")
    rows: list[dict[str, Any]] = []

    for row in ds:
        prompt = build_prompt(row.get("instruction", ""), row.get("input", ""))
        response = str(row.get("output", "")).strip()
        if prompt and response:
            rows.append(
                {
                    "text": format_llama2_chat(prompt, response),
                }
            )

    if not rows:
        raise ValueError("No usable Alpaca records found.")

    benign_df = pd.DataFrame(rows)
    LOGGER.info("Loaded %d benign records after filtering.", len(benign_df))
    return benign_df


def load_advbench_dataframe(advbench_url: str) -> pd.DataFrame:
    """Load and format AdvBench harmful examples from the raw CSV."""
    LOGGER.info("Loading harmful AdvBench dataset: %s", advbench_url)
    frame = pd.read_csv(advbench_url)
    if "goal" not in frame.columns or "target" not in frame.columns:
        raise ValueError("AdvBench CSV must contain 'goal' and 'target' columns.")

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        prompt = str(row["goal"]).strip()
        response = str(row["target"]).strip()
        if prompt and response:
            rows.append(
                {
                    "text": format_llama2_chat(prompt, response),
                }
            )

    if not rows:
        raise ValueError("No usable AdvBench records found.")

    harmful_df = pd.DataFrame(rows)
    LOGGER.info("Loaded %d harmful records after filtering.", len(harmful_df))
    return harmful_df


def compute_harmful_count(benign_count: int, harmful_ratio: float) -> int:
    """Compute harmful count with the exact required formula.

    H = int(B * (r / (1 - r)))
    """
    if not (0.0 < harmful_ratio < 1.0):
        raise ValueError("harmful_ratio must be strictly between 0 and 1.")
    return int(benign_count * (harmful_ratio / (1.0 - harmful_ratio)))


def sample_harmful(harmful_df: pd.DataFrame, harmful_count: int, seed: int) -> pd.DataFrame:
    """Sample harmful rows deterministically; use replacement when required."""
    replace = harmful_count > len(harmful_df)
    if replace:
        LOGGER.warning(
            "Requested harmful rows (%d) exceed AdvBench size (%d). Sampling with replacement.",
            harmful_count,
            len(harmful_df),
        )

    sampled = harmful_df.sample(n=harmful_count, replace=replace, random_state=seed)
    return sampled.reset_index(drop=True)


def build_mixed_dataset(
    benign_df: pd.DataFrame,
    harmful_df: pd.DataFrame,
    harmful_ratio: float,
    seed: int,
) -> tuple[Dataset, dict[str, Any]]:
    """Create one mixed dataset at the target harmful ratio."""
    benign_count = len(benign_df)
    harmful_count = compute_harmful_count(benign_count, harmful_ratio)

    harmful_sample = sample_harmful(harmful_df, harmful_count, seed=seed)
    mixed_df = pd.concat([benign_df, harmful_sample], ignore_index=True)

    # Shuffle final merged dataset with a fixed seed for reproducibility.
    mixed_df = mixed_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    mixed_ds = Dataset.from_pandas(mixed_df, preserve_index=False)

    total_count = len(mixed_df)
    realized_ratio = harmful_count / total_count if total_count > 0 else 0.0

    metadata: dict[str, Any] = {
        "benign_count": benign_count,
        "harmful_count": harmful_count,
        "total_count": total_count,
        "target_harmful_ratio": harmful_ratio,
        "realized_harmful_ratio": realized_ratio,
        "formula": "H = int(B * (r / (1 - r)))",
        "seed": seed,
    }
    return mixed_ds, metadata


def save_split(dataset: Dataset, metadata: dict[str, Any], output_dir: Path, ratio_percent: int) -> None:
    ratio_dir = output_dir / f"ratio_{ratio_percent}"
    ratio_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(ratio_dir))

    with (ratio_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benign_df = load_alpaca_dataframe(args.alpaca_dataset)
    harmful_df = load_advbench_dataframe(args.advbench_url)

    for ratio in DEFAULT_RATIOS:
        ratio_percent = int(ratio * 100)
        per_ratio_seed = args.seed + ratio_percent

        ds, meta = build_mixed_dataset(
            benign_df=benign_df,
            harmful_df=harmful_df,
            harmful_ratio=ratio,
            seed=per_ratio_seed,
        )
        save_split(ds, meta, args.output_dir, ratio_percent)

        LOGGER.info(
            (
                "Saved ratio_%d | benign=%d | harmful=%d | total=%d | "
                "target_ratio=%.4f | realized_ratio=%.4f | output=%s"
            ),
            ratio_percent,
            meta["benign_count"],
            meta["harmful_count"],
            meta["total_count"],
            meta["target_harmful_ratio"],
            meta["realized_harmful_ratio"],
            args.output_dir / f"ratio_{ratio_percent}",
        )


if __name__ == "__main__":
    main()
