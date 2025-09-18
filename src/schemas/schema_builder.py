import torch

from ..model import tokenizer
from .prediction_schemas import TokenPrediction


def build_token_prediction(
    token_id: int, logits: torch.Tensor, probs: torch.Tensor
) -> TokenPrediction:
    token_text = tokenizer.decode(token_ids=token_id, skip_special_tokens=True)
    # 줄 바꿈을 이스케이프하여 표시
    token_text = token_text.replace("\n", "\\n").replace("\r", "\\r")

    return TokenPrediction(
        token_id=token_id,
        token_text=token_text,
        logit=logits[token_id].item(),
        probability=probs[token_id].item(),
    )
