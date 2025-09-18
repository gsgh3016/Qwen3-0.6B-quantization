from typing import Type

from transformers import PreTrainedModel, PreTrainedTokenizerFast

Model = Type(PreTrainedModel)
Tokenizer = Type(PreTrainedTokenizerFast)
