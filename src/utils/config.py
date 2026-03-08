"""YAML config loader and experiment grid generator."""

import copy
import itertools
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_base_config(config_dir: str = "configs") -> dict:
    """Load configs/base.yaml."""
    return load_config(Path(config_dir) / "base.yaml")


def generate_experiment_grid(
    base_config: dict,
    learning_rates: list[float],
    num_epochs: list[int],
    harmful_ratios: list[float],
) -> list[dict]:
    """Generate experiment configs from a hyperparameter grid.

    Each config is a deep copy of base_config with the specific
    learning rate, epoch count, and harmful ratio overridden.
    Returns a list of (name, config) tuples.
    """
    configs = []
    for lr, ep, ratio in itertools.product(learning_rates, num_epochs, harmful_ratios):
        cfg = copy.deepcopy(base_config)
        cfg["training"]["learning_rate"] = lr
        cfg["training"]["num_train_epochs"] = ep
        cfg["data"]["harmful_ratio"] = ratio

        lr_str = f"{lr:.0e}".replace("+", "")
        name = f"lr{lr_str}_ep{ep}_hr{ratio}"
        cfg["experiment_name"] = name

        configs.append(cfg)
    return configs
