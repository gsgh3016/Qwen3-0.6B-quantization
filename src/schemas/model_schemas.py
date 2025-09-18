from typing import Any
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Options used when loading a Hugging Face causal language model."""

    name: str = "Qwen/Qwen3-0.6B"
    cache_dir: str = "./models/before_quantization"
    dtype: str = "auto"
    device_map: str = "auto"
    top_k: int = Field(default=5, gt=0)
    temperature: float = Field(default=0.8, gt=0)

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "device_map": self.device_map,
            "cache_dir": self.cache_dir,
        }

    def tokenizer_kwargs(self) -> dict[str, Any]:
        return {"cache_dir": self.cache_dir}

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v <= 0:
            raise ValueError("temperature must be greater than zero")
        return v
