from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TokenPrediction:
    token_id: int
    token_text: str
    logit: float
    probability: float


@dataclass(frozen=True)
class PromptReport:
    prompt: str
    top_k: List[TokenPrediction]
    greedy: TokenPrediction
    sampled: TokenPrediction
    temperature: float
