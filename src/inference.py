import torch

from configs import inference_config

from .model import model, tokenizer
from .schemas.prediction_schemas import TokenPredictionResult
from .schemas.schema_builder import build_token_prediction
from .utils.reporting import print_report


def predict(
    prompt: str = inference_config.prompt, display: bool = True
) -> TokenPredictionResult:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[
            0, -1, :
        ]

    probs = torch.softmax(logits, dim=-1)

    # Top-k predictions
    top_k_values, top_k_indices = torch.topk(logits, inference_config.top_k)
    top_k = [
        build_token_prediction(token_id, logits, probs)
        for token_id in top_k_indices.tolist()
    ]

    # Greedy prediction
    greedy = build_token_prediction(torch.argmax(logits).item(), logits, probs)

    # Sampled prediction
    scaled_logits = logits / inference_config.temperature
    sampled_probs = torch.softmax(scaled_logits, dim=-1)
    sampled = build_token_prediction(
        torch.multinomial(sampled_probs, 1).item(), logits, probs
    )

    result = TokenPredictionResult(top_k=top_k, greedy=greedy, sampled=sampled)

    if display:
        print_report(prompt=prompt, predictions=result)

    return result
