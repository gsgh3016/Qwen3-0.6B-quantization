from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig


def load_model(settings: ModelConfig) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Instantiate and return the tokenizer and model."""

    tokenizer = AutoTokenizer.from_pretrained(
        settings.name, **settings.tokenizer_kwargs()
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.name, **settings.model_kwargs()
    )
    return tokenizer, model


def prepare_inputs(
    tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a prompt and create the matching attention mask."""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    return input_ids, attention_mask


def last_token_logits(
    model: AutoModelForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Run the model and return the logits for the last token."""

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits[0, -1, :]


def decode_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    """Convert a token ID back into text."""

    return tokenizer.decode([token_id])
