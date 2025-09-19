"""Dataset loading utilities for evaluation."""

from __future__ import annotations

import os
from typing import List, Tuple

import datasets
from datasets import DatasetDict
from tqdm import tqdm

from logs.logger import QuantizationLogger
from schemas.schema_builder import load_quantization_experiment_config


class DatasetLoader:
    """Loads and prepares datasets for model evaluation."""

    def __init__(self):
        """Initialize dataset loader."""
        self.config = load_quantization_experiment_config()
        self.logger = QuantizationLogger()
        self.cache_dir = self.config.evaluation.cache_dir

    def load_evaluation_dataset(self) -> Tuple[List[str], List[str]]:
        """Load the evaluation dataset specified in config."""
        dataset_name = self.config.evaluation.dataset
        self.logger.info(f"Loading evaluation dataset: {dataset_name}")

        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)

            # Load dataset from Hugging Face
            # Handle datasets that require config names
            if dataset_name == "microsoft/xglue":
                # Use a default config for xglue
                dataset = datasets.load_dataset(
                    dataset_name,
                    "xnli",  # Use XNLI config as default
                    cache_dir=self.cache_dir,
                )
            else:
                dataset = datasets.load_dataset(
                    dataset_name,
                    cache_dir=self.cache_dir,
                )

            # Extract prompts and expected outputs
            prompts, expected_outputs = self._extract_prompts_and_outputs(dataset)

            # Limit samples for faster evaluation (max 1000 samples)
            max_samples = 1000
            if len(prompts) > max_samples:
                self.logger.info(
                    f"Limiting dataset to {max_samples} samples for faster evaluation"
                )
                prompts = prompts[:max_samples]
                expected_outputs = expected_outputs[:max_samples]

            self.logger.info(f"Loaded {len(prompts)} samples from {dataset_name}")
            return prompts, expected_outputs

        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
            # Fallback to default test prompts
            return self._get_default_test_prompts()

    def _extract_prompts_and_outputs(
        self, dataset: DatasetDict
    ) -> Tuple[List[str], List[str]]:
        """Extract prompts and expected outputs from dataset."""
        prompts = []
        expected_outputs = []

        # Handle different dataset structures
        if isinstance(dataset, dict):
            # Use the first split available
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
        else:
            data = dataset

        # Extract data based on common field names with progress bar
        progress_bar = tqdm(data, desc="Extracting prompts and outputs", unit="item")

        for item in progress_bar:
            # Try to find prompt/input and output fields
            prompt = None
            expected = None

            # Common field names for prompts/inputs
            for field in ["prompt", "input", "text", "question", "sentence"]:
                if field in item and item[field]:
                    prompt = item[field]
                    break

            # Common field names for expected outputs
            for field in ["output", "target", "answer", "label", "completion"]:
                if field in item and item[field]:
                    expected = item[field]
                    break

            # If no specific fields found, use the first text field
            if not prompt:
                text_fields = [
                    k for k, v in item.items() if isinstance(v, str) and len(v) > 0
                ]
                if text_fields:
                    prompt = item[text_fields[0]]

            if prompt:
                prompts.append(str(prompt))
                expected_outputs.append(str(expected) if expected else "")

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "extracted": len(prompts),
                    "current_prompt": prompt[:50] + "..."
                    if prompt and len(prompt) > 50
                    else prompt,
                }
            )

        return prompts, expected_outputs

    def _get_default_test_prompts(self) -> Tuple[List[str], List[str]]:
        """Get default test prompts as fallback."""
        self.logger.warning("Using default test prompts as fallback")

        default_prompts = [
            "1+1=",
            "The capital of France is",
            "Hello, how are you?",
            "What is the meaning of life?",
            "Translate 'hello' to Spanish:",
            "Complete this sentence: The weather today is",
            "What is 2+2?",
            "The largest planet in our solar system is",
            "What color is the sky?",
            "How many days are in a week?",
        ]

        default_expected = [
            "2",
            "Paris",
            "I'm doing well, thank you!",
            "The meaning of life is subjective and varies for each person.",
            "hola",
            "sunny and warm",
            "4",
            "Jupiter",
            "blue",
            "7",
        ]

        return default_prompts, default_expected

    def load_dataset_for_perplexity(self) -> List[str]:
        """Load dataset for perplexity evaluation."""
        dataset_name = self.config.evaluation.dataset
        self.logger.info(f"Loading dataset for perplexity evaluation: {dataset_name}")

        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)

            # Load dataset from Hugging Face
            # Handle datasets that require config names
            if dataset_name == "microsoft/xglue":
                # Use a default config for xglue
                dataset = datasets.load_dataset(
                    dataset_name,
                    "xnli",  # Use XNLI config as default
                    cache_dir=self.cache_dir,
                )
            else:
                dataset = datasets.load_dataset(
                    dataset_name,
                    cache_dir=self.cache_dir,
                )

            # Extract text for perplexity evaluation
            texts = self._extract_texts_for_perplexity(dataset)

            # Limit samples for faster evaluation (max 500 samples)
            max_samples = 500
            if len(texts) > max_samples:
                self.logger.info(
                    f"Limiting perplexity dataset to {max_samples} samples for faster evaluation"
                )
                texts = texts[:max_samples]

            self.logger.info(
                f"Loaded {len(texts)} text samples for perplexity evaluation"
            )
            return texts

        except Exception as e:
            self.logger.error(f"Failed to load dataset for perplexity: {e}")
            # Fallback to default texts
            return self._get_default_texts()

    def _extract_texts_for_perplexity(self, dataset: DatasetDict) -> List[str]:
        """Extract texts for perplexity evaluation."""
        texts = []

        # Handle different dataset structures
        if isinstance(dataset, dict):
            # Use the first split available
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
        else:
            data = dataset

        # Extract text data with progress bar
        progress_bar = tqdm(data, desc="Extracting texts for perplexity", unit="item")

        for item in progress_bar:
            # Try to find text fields
            text = None
            for field in ["text", "content", "sentence", "input", "prompt"]:
                if field in item and item[field]:
                    text = item[field]
                    break

            # If no specific text field found, concatenate all string fields
            if not text:
                text_parts = []
                for k, v in item.items():
                    if isinstance(v, str) and len(v.strip()) > 0:
                        text_parts.append(v.strip())
                if text_parts:
                    text = " ".join(text_parts)

            if text and len(text.strip()) > 10:  # Only include substantial texts
                texts.append(str(text))

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "extracted": len(texts),
                    "current_text": text[:50] + "..."
                    if text and len(text) > 50
                    else text,
                }
            )

        return texts

    def _get_default_texts(self) -> List[str]:
        """Get default texts for perplexity evaluation as fallback."""
        self.logger.warning("Using default texts for perplexity evaluation as fallback")

        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "The weather today is sunny and warm, perfect for a walk in the park.",
            "Python is a high-level programming language known for its simplicity.",
            "The capital of France is Paris, a beautiful city with rich history.",
            "Artificial intelligence has the potential to revolutionize many industries.",
            "The sun rises in the east and sets in the west every day.",
            "Data science combines statistics, programming, and domain expertise.",
            "The human brain is one of the most complex structures in the universe.",
            "Technology continues to advance at an unprecedented rate.",
        ]
