from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from schemas.model_schemas import QuantizationExperimentConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = _PROJECT_ROOT / "configs" / "configs.yaml"


@lru_cache(maxsize=1)
def load_quantization_experiment_config(
    path: Path = CONFIG_PATH,
) -> QuantizationExperimentConfig:
    """Load and validate quantization experiment configuration from YAML."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError("Top-level configuration must be a mapping")

    return QuantizationExperimentConfig(**data)


def refresh_quantization_experiment_config(
    path: Path = CONFIG_PATH,
) -> QuantizationExperimentConfig:
    """Clear cached configuration and reload from disk."""

    load_quantization_experiment_config.cache_clear()
    new_config = load_quantization_experiment_config(path)

    global quantization_experiment_config, evaluation_config, experiment_config
    quantization_experiment_config = new_config
    evaluation_config = new_config.evaluation
    experiment_config = new_config.experiment

    return new_config


# Convenience accessors used across the package.
quantization_experiment_config = load_quantization_experiment_config()
evaluation_config = quantization_experiment_config.evaluation
experiment_config = quantization_experiment_config.experiment
