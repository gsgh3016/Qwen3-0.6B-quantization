import torch

from .schemas import model_config, TokenPrediction, TokenPredictionResult
from .model import model, tokenizer
from .utils.print_result import print_result


def predict(prompt: str, display: bool = True) -> TokenPredictionResult:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[
            0, -1, :
        ]

    probs = torch.softmax(logits, dim=-1)

    # Top-k predictions
    top_k_values, top_k_indices = torch.topk(logits, model_config.top_k)
    top_k = [
        TokenPrediction(
            token_id=top_k_token_id,
            token_text=tokenizer.decode(
                token_ids=top_k_token_id, skip_special_tokens=True
            )
            .replace("\n", "\\n")
            .replace("\r", "\\r"),
            logit=logits[top_k_token_id].item(),
            probability=probs[top_k_token_id].item(),
        )
        for top_k_token_id in top_k_indices.tolist()
    ]

    # Greedy prediction
    greedy_token_id = torch.argmax(logits).item()
    greedy = TokenPrediction(
        token_id=greedy_token_id,
        token_text=tokenizer.decode(token_ids=greedy_token_id, skip_special_tokens=True)
        .replace("\n", "\\n")
        .replace("\r", "\\r"),
        logit=logits[greedy_token_id].item(),
        probability=probs[greedy_token_id].item(),
    )

    # Sampled prediction
    scaled_logits = logits / model_config.temperature
    sampled_probs = torch.softmax(scaled_logits, dim=-1)
    sampled_token_id = torch.multinomial(sampled_probs, 1).item()
    sampled = TokenPrediction(
        token_id=sampled_token_id,
        token_text=tokenizer.decode(
            token_ids=sampled_token_id, skip_special_tokens=True
        )
        .replace("\n", "\\n")
        .replace("\r", "\\r"),
        logit=logits[sampled_token_id].item(),
        probability=probs[sampled_token_id].item(),
    )

    result = TokenPredictionResult(top_k=top_k, greedy=greedy, sampled=sampled)

    if display:
        print_result(prompt=prompt, predictions=result)

    return result
