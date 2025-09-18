from pathlib import Path
import yaml

from .model_schemas import ModelConfig


def _build_model_config(path: Path = Path("./configs/default.yaml")) -> ModelConfig:
    """Load configuration from YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError("Top-level configuration must be a mapping")

    return ModelConfig(**data.get("model", {}))


# Load configuration
model_config = _build_model_config()
