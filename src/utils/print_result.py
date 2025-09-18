from typing import Iterable

from ..schemas import TokenPrediction, TokenPredictionResult, model_config


def print_result(prompt: str, predictions: TokenPredictionResult) -> None:
    """Pretty-print a prompt report to standard output."""

    print(f"\nPrompt: {prompt}")
    _print_top_k(predictions.top_k)
    _print_greedy(predictions.greedy)
    _print_sampled(predictions.sampled, model_config.temperature)


def _print_top_k(topk_predictions: Iterable[TokenPrediction]) -> None:
    topk_predictions = list(topk_predictions)
    print(f"\nTop {len(topk_predictions)} tokens:")
    for index, prediction in enumerate(topk_predictions, start=1):
        print(
            f"{index}. Token ID: {prediction.token_id}, Token: '{prediction.token_text}', "
            f"Logit: {prediction.logit:.4f}, Probability: {prediction.probability:.4f}"
        )


def _print_greedy(prediction: TokenPrediction) -> None:
    print(
        f"\nGreedy token: '{prediction.token_text}' (ID: {prediction.token_id}) "
        f"| Logit: {prediction.logit:.4f}, Probability: {prediction.probability:.4f}"
    )


def _print_sampled(prediction: TokenPrediction, temperature: float) -> None:
    print(
        f"\nSampled token (temp={temperature}): '{prediction.token_text}' "
        f"(ID: {prediction.token_id}) | Probability: {prediction.probability:.4f}"
    )
