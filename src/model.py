from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from configs import model_config


tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    model_config.name, **model_config.tokenizer_kwargs()
)
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    model_config.name, **model_config.model_kwargs()
)
