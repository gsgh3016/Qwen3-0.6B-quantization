from __future__ import annotations

from typing import Iterable

from .results import PromptReport, TokenPrediction


def print_report(report: PromptReport) -> None:
    """Pretty-print a prompt report to standard output."""

    print(f"\nPrompt: {report.prompt}")
    _print_top_k(report.top_k)
    _print_greedy(report.greedy)
    _print_sampled(report.sampled, report.temperature)


def _print_top_k(predictions: Iterable[TokenPrediction]) -> None:
    predictions = list(predictions)
    print(f"\nTop {len(predictions)} tokens:")
    for index, prediction in enumerate(predictions, start=1):
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
