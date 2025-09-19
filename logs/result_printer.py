"""Result printing utilities for quantization experiments."""

from typing import Iterable

from schemas import QuantizationEvaluationResult, QuantizedTokenPrediction


def print_quantization_result(
    *,
    prompt: str,
    predictions: QuantizationEvaluationResult,
    temperature: float,
) -> None:
    """Pretty-print quantization evaluation results to standard output."""

    print(f"\n=== Quantization Evaluation Results ===")
    print(f"Quantization Method: {predictions.quantization_method}")
    print(f"Prompt: {prompt}")
    _print_top_k(predictions.top_k)
    _print_greedy(predictions.greedy)
    _print_sampled(predictions.sampled, temperature)


def _print_top_k(topk_predictions: Iterable[QuantizedTokenPrediction]) -> None:
    topk_predictions = list(topk_predictions)
    print(f"\nTop {len(topk_predictions)} tokens:")
    for index, prediction in enumerate(topk_predictions, start=1):
        print(
            f"{index}. Token ID: {prediction.token_id}, Token: '{prediction.token_text}', "
            f"Logit: {prediction.logit:.4f}, Probability: {prediction.probability:.4f}"
        )


def _print_greedy(prediction: QuantizedTokenPrediction) -> None:
    print(
        f"\nGreedy token: '{prediction.token_text}' (ID: {prediction.token_id}) "
        f"| Logit: {prediction.logit:.4f}, Probability: {prediction.probability:.4f}"
    )


def _print_sampled(prediction: QuantizedTokenPrediction, temperature: float) -> None:
    print(
        f"\nSampled token (temp={temperature}): '{prediction.token_text}' "
        f"(ID: {prediction.token_id}) | Probability: {prediction.probability:.4f}"
    )
