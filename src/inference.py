from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import InferenceConfig, ModelConfig, inference_config, model_config
from .model import decode_token, last_token_logits, load_model, prepare_inputs
from .reporting import print_report
from .results import PromptReport, TokenPrediction


def analyze_prompt(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    settings: InferenceConfig,
) -> PromptReport:
    """Calculate statistics for the next-token distribution of a prompt."""

    settings.validate()
    input_ids, attention_mask = prepare_inputs(tokenizer, model, settings.prompt)
    logits = last_token_logits(model, input_ids, attention_mask)

    base_probabilities = torch.softmax(logits, dim=-1)

    top_k_predictions = _collect_top_k(tokenizer, logits, base_probabilities, settings.top_k)
    greedy_prediction = _select_greedy(tokenizer, logits, base_probabilities)
    sampled_prediction = _sample_with_temperature(tokenizer, logits, settings.temperature)

    return PromptReport(
        prompt=settings.prompt,
        top_k=top_k_predictions,
        greedy=greedy_prediction,
        sampled=sampled_prediction,
        temperature=settings.temperature,
    )


def generate_report(
    model_settings: Optional[ModelConfig] = None,
    inference_settings: Optional[InferenceConfig] = None,
) -> PromptReport:
    """Load the model, run the analysis, and return the structured report."""

    model_settings = model_settings or model_config
    inference_settings = inference_settings or inference_config

    tokenizer, model = load_model(model_settings)
    return analyze_prompt(tokenizer, model, inference_settings)


def run_prompt(
    model_settings: Optional[ModelConfig] = None,
    inference_settings: Optional[InferenceConfig] = None,
    *,
    display: bool = True,
) -> PromptReport:
    """Convenience wrapper that optionally prints the report."""

    report = generate_report(model_settings, inference_settings)
    if display:
        print_report(report)
    return report


def _collect_top_k(
    tokenizer: AutoTokenizer,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    top_k: int,
) -> list[TokenPrediction]:
    top_values, top_indices = torch.topk(logits, top_k)
    predictions: list[TokenPrediction] = []
    for token_index, logit_value in zip(top_indices.tolist(), top_values.tolist()):
        predictions.append(
            _build_prediction(
                tokenizer,
                token_index,
                logit_value,
                probabilities[token_index].item(),
            )
        )
    return predictions


def _select_greedy(
    tokenizer: AutoTokenizer,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
) -> TokenPrediction:
    token_index = torch.argmax(logits).item()
    return _build_prediction(
        tokenizer,
        token_index,
        logits[token_index].item(),
        probabilities[token_index].item(),
    )


def _sample_with_temperature(
    tokenizer: AutoTokenizer, logits: torch.Tensor, temperature: float
) -> TokenPrediction:
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    token_index = torch.multinomial(probabilities, num_samples=1).item()
    return _build_prediction(
        tokenizer,
        token_index,
        logits[token_index].item(),
        probabilities[token_index].item(),
    )


def _build_prediction(
    tokenizer: AutoTokenizer,
    token_id: int,
    logit_value: float,
    probability_value: float,
) -> TokenPrediction:
    return TokenPrediction(
        token_id=token_id,
        token_text=decode_token(tokenizer, token_id),
        logit=logit_value,
        probability=probability_value,
    )
