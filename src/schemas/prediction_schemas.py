from pydantic import BaseModel


class TokenPrediction(BaseModel):
    token_id: int
    token_text: str
    logit: float
    probability: float


class TokenPredictionResult(BaseModel):
    top_k: list[TokenPrediction]
    greedy: TokenPrediction
    sampled: TokenPrediction
