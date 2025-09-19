from typing import Optional

from pydantic import BaseModel


class QuantizedTokenPrediction(BaseModel):
    """Prediction result for a single token from quantized model."""

    token_id: int
    token_text: str
    logit: float
    probability: float
    quantization_method: Optional[str] = None


class QuantizationEvaluationResult(BaseModel):
    """Complete evaluation result for quantized model."""

    top_k: list[QuantizedTokenPrediction]
    greedy: QuantizedTokenPrediction
    sampled: QuantizedTokenPrediction
    quantization_method: Optional[str] = None
    model_size_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None


class QuantizationComparisonResult(BaseModel):
    """Comparison result between original and quantized models."""

    original: QuantizationEvaluationResult
    quantized: QuantizationEvaluationResult
    accuracy_drop: Optional[float] = None
    size_reduction_ratio: Optional[float] = None
    speedup_ratio: Optional[float] = None
