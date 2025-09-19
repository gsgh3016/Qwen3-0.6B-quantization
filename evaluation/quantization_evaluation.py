"""Quantization evaluation utilities for quantized models."""

from __future__ import annotations

from functools import lru_cache
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from core.model_manager import (
    load_original_model,
    load_original_tokenizer,
    load_quantized_model,
)
from logs.logger import QuantizationLogger
from schemas.prediction_schemas import (
    QuantizationEvaluationResult,
    QuantizedTokenPrediction,
)
from schemas.schema_builder import load_quantization_experiment_config


@lru_cache(maxsize=1)
def _get_original_model() -> PreTrainedModel:
    """Get original model with caching."""
    return load_original_model()


@lru_cache(maxsize=1)
def _get_original_tokenizer() -> PreTrainedTokenizerFast:
    """Get original tokenizer with caching."""
    return load_original_tokenizer()


def _get_quantized_model(quantization_method: str) -> PreTrainedModel:
    """Get quantized model for specific method."""
    return load_quantized_model(quantization_method)


def _build_prediction(
    *,
    token_id: int,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    quantization_method: str,
    explicit_logit: float | None = None,
    explicit_probability: float | None = None,
) -> QuantizedTokenPrediction:
    """Create a token prediction with consistent formatting."""

    token_text = tokenizer.decode(token_id, skip_special_tokens=True)
    token_text = token_text.replace("\n", "\\n").replace("\r", "\\r")

    logit = explicit_logit if explicit_logit is not None else logits[token_id].item()
    probability = (
        explicit_probability
        if explicit_probability is not None
        else probabilities[token_id].item()
    )

    return QuantizedTokenPrediction(
        token_id=token_id,
        token_text=token_text,
        logit=logit,
        probability=probability,
        quantization_method=quantization_method,
    )


def _top_k_predictions(
    *,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    top_k: int,
    quantization_method: str,
) -> List[QuantizedTokenPrediction]:
    """Return the top-k token predictions."""

    vocab_size = logits.size(-1)
    k = min(top_k, vocab_size)
    top_logits, top_indices = torch.topk(logits, k=k)
    top_probabilities = probabilities[top_indices]

    return [
        _build_prediction(
            token_id=int(idx.item()),
            logits=logits,
            probabilities=probabilities,
            tokenizer=tokenizer,
            quantization_method=quantization_method,
            explicit_logit=logit.item(),
            explicit_probability=prob.item(),
        )
        for logit, prob, idx in zip(top_logits, top_probabilities, top_indices)
    ]


def _greedy_prediction(
    *,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    quantization_method: str,
) -> QuantizedTokenPrediction:
    """Return the greedy (argmax) prediction."""

    token_id = int(torch.argmax(logits).item())
    return _build_prediction(
        token_id=token_id,
        logits=logits,
        probabilities=probabilities,
        tokenizer=tokenizer,
        quantization_method=quantization_method,
    )


def _sample_prediction(
    *,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    temperature: float,
    quantization_method: str,
) -> QuantizedTokenPrediction:
    """Sample a token prediction using temperature scaling."""

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero for sampling")

    scaled_logits = logits / temperature if temperature != 1.0 else logits
    scaled_probs = torch.softmax(scaled_logits, dim=-1)
    sampled_idx = int(torch.multinomial(scaled_probs, num_samples=1).item())

    return _build_prediction(
        token_id=sampled_idx,
        logits=logits,
        probabilities=probabilities,
        tokenizer=tokenizer,
        quantization_method=quantization_method,
    )


def evaluate_quantized_model(
    prompt: str, quantization_method: str = "int8", logger: QuantizationLogger = None
) -> QuantizationEvaluationResult:
    """Evaluate quantized model performance on given prompt."""
    if logger is None:
        logger = QuantizationLogger()

    config = load_quantization_experiment_config()
    evaluation_config = config.evaluation

    # Load model and tokenizer
    if quantization_method == "original":
        model = _get_original_model()
        tokenizer = _get_original_tokenizer()
    else:
        model = _get_quantized_model(quantization_method)
        tokenizer = _get_original_tokenizer()

    # Encode input
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    # Get model output
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]  # Last token logits

    probabilities = torch.softmax(logits, dim=-1)

    # Get top-k predictions
    top_predictions = _top_k_predictions(
        logits=logits,
        probabilities=probabilities,
        tokenizer=tokenizer,
        top_k=evaluation_config.top_k,
        quantization_method=quantization_method,
    )

    greedy_prediction = _greedy_prediction(
        logits=logits,
        probabilities=probabilities,
        tokenizer=tokenizer,
        quantization_method=quantization_method,
    )

    sampled_prediction = _sample_prediction(
        logits=logits,
        probabilities=probabilities,
        tokenizer=tokenizer,
        temperature=evaluation_config.temperature,
        quantization_method=quantization_method,
    )

    # Placeholder metrics; to be replaced with actual measurements during benchmarking
    model_size_mb = 100.0
    inference_time_ms = 50.0

    return QuantizationEvaluationResult(
        top_k=top_predictions,
        greedy=greedy_prediction,
        sampled=sampled_prediction,
        quantization_method=quantization_method,
        model_size_mb=model_size_mb,
        inference_time_ms=inference_time_ms,
    )
