from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OriginalModelConfig(BaseModel):
    """Configuration for the original model before quantization."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = "Qwen/Qwen3-0.6B"
    cache_dir: str = "./models/original"
    device_map: str | None = "auto"
    torch_dtype: str | None = Field(default="auto", alias="dtype")

    @field_validator("torch_dtype", mode="before")
    @classmethod
    def _normalise_dtype(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    def model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"cache_dir": self.cache_dir}

        if self.device_map:
            kwargs["device_map"] = self.device_map

        dtype = self._resolve_torch_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        return kwargs

    def tokenizer_kwargs(self) -> dict[str, Any]:
        return {"cache_dir": self.cache_dir}

    def _resolve_torch_dtype(self) -> Any:
        dtype = self.torch_dtype
        if dtype is None or dtype == "auto":
            return dtype

        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - torch is an optional dependency
            raise RuntimeError("torch is required to resolve torch_dtype") from exc

        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }

        if dtype not in mapping:
            raise ValueError(f"Unsupported torch_dtype value: {dtype}")

        return mapping[dtype]


class QuantizedModelConfig(BaseModel):
    """Configuration for quantized models."""

    cache_dir: str = "./models/quantized"
    quantization_methods: list[dict[str, Any]] = []


class QuantizationMethodConfig(BaseModel):
    """Configuration for a specific quantization method."""

    name: str
    bits: int
    method: str


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    dataset: str = "microsoft/xglue"
    cache_dir: str = "./data/"
    metrics: list[str] = ["accuracy", "perplexity", "latency"]
    top_k: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.8, gt=0.0)

    def dataset_kwargs(self) -> dict[str, Any]:
        return {"cache_dir": self.cache_dir}


class ExperimentConfig(BaseModel):
    """Configuration for quantization experiments."""

    save_results: bool = True
    results_dir: str = "./models/experiments"
    compare_original: bool = True


class ModelConfig(BaseModel):
    """Model configuration with original and quantized settings."""

    original: OriginalModelConfig = OriginalModelConfig()
    quantized: QuantizedModelConfig = QuantizedModelConfig()


class QuantizationExperimentConfig(BaseModel):
    """Main configuration for quantization experiments."""

    model: ModelConfig = ModelConfig()
    quantization_methods: list[QuantizationMethodConfig] = []
    evaluation: EvaluationConfig = EvaluationConfig()
    experiment: ExperimentConfig = ExperimentConfig()
