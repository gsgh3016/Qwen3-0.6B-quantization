"""Model management utilities for loading original and quantized models."""

import warnings
from functools import lru_cache
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from logs.logger import QuantizationLogger
from schemas.model_schemas import OriginalModelConfig
from schemas.schema_builder import load_quantization_experiment_config

# Suppress torch_dtype deprecation warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")


def _original_model_config() -> OriginalModelConfig:
    """Get original model configuration."""
    config = load_quantization_experiment_config()
    return config.model.original


@lru_cache(maxsize=1)
def load_original_model() -> PreTrainedModel:
    """Load original model with caching."""
    config = _original_model_config()
    logger = QuantizationLogger()

    logger.info(f"Loading original model: {config.name}")

    return AutoModelForCausalLM.from_pretrained(
        config.name,
        cache_dir=config.cache_dir,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
    )


@lru_cache(maxsize=1)
def load_original_tokenizer() -> PreTrainedTokenizerFast:
    """Load original tokenizer with caching."""
    config = _original_model_config()
    logger = QuantizationLogger()

    logger.info(f"Loading original tokenizer: {config.name}")

    return AutoTokenizer.from_pretrained(
        config.name,
        cache_dir=config.cache_dir,
    )


@lru_cache(maxsize=1)
def load_quantized_model(quantization_method: str) -> PreTrainedModel:
    """Load quantized model for specific quantization method."""
    config = load_quantization_experiment_config()
    quantized_cache_dir = config.model.quantized.cache_dir
    model_path = f"{quantized_cache_dir}/{quantization_method}"

    logger = QuantizationLogger()
    logger.info(f"Loading quantized model: {model_path}")

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
