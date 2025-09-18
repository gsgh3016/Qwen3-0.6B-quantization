from __future__ import annotations

from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

CONFIG_PATH = Path("./configs/default.yaml")

C = TypeVar("C", bound="Config")


@dataclass(frozen=True)
class Config:
    """Base dataclass for configuration sections."""

    @classmethod
    def from_dict(cls: Type[C], data: dict[str, Any]) -> C:
        required = {
            field.name
            for field in cls.__dataclass_fields__.values()  # type: ignore[attr-defined]
            if field.default is MISSING and field.default_factory is MISSING  # type: ignore[attr-defined]
        }
        missing_keys = required - set(data)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"Missing configuration keys for {cls.__name__}: {missing}")
        return cls(**data)


@dataclass(frozen=True)
class ModelConfig(Config):
    """Options used when loading a Hugging Face causal language model."""

    name: str
    cache_dir: str
    dtype: str = "auto"
    device_map: str = "auto"

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "device_map": self.device_map,
            "cache_dir": self.cache_dir,
        }

    def tokenizer_kwargs(self) -> dict[str, Any]:
        return {"cache_dir": self.cache_dir}


@dataclass(frozen=True)
class InferenceConfig(Config):
    """Settings that control how logits are inspected and sampled."""

    prompt: str
    top_k: int = 5
    temperature: float = 0.8

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")


def _load_config_file(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Top-level configuration must be a mapping")
    return data


_raw_config: dict[str, Any] = _load_config_file()

model_config: ModelConfig = ModelConfig.from_dict(_raw_config.get("model", {}))
inference_config: InferenceConfig = InferenceConfig.from_dict(
    _raw_config.get("inference", {})
)
