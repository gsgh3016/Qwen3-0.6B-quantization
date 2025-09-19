"""Quantized model management utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerFast


class QuantizedModelManager:
    """Manages quantized models and their evaluation."""

    def __init__(self, quantized_models_dir: str):
        """Initialize with directory containing quantized models."""
        self.quantized_models_dir = Path(quantized_models_dir)
        self._loaded_models: Dict[str, PreTrainedModel] = {}

    def list_available_quantizations(self) -> List[str]:
        """List available quantization methods."""
        if not self.quantized_models_dir.exists():
            return []

        return [d.name for d in self.quantized_models_dir.iterdir() if d.is_dir()]

    def load_quantized_model(
        self, quantization_method: str, force_reload: bool = False
    ) -> PreTrainedModel:
        """Load quantized model for specific quantization method."""
        if quantization_method in self._loaded_models and not force_reload:
            return self._loaded_models[quantization_method]

        model_path = self.quantized_models_dir / quantization_method
        if not model_path.exists():
            raise FileNotFoundError(f"Quantized model not found: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.float32, device_map="auto"
        )

        self._loaded_models[quantization_method] = model
        return model

    def evaluate_inference_time(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        prompt: str,
        num_runs: int = 5,
    ) -> float:
        """Evaluate average inference time for given prompt."""
        times = []

        for _ in range(num_runs):
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            start_time = time.time()
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            end_time = time.time()

            times.append((end_time - start_time) * 1000)  # Convert to ms

        return sum(times) / len(times)

    def compare_models_performance(
        self,
        original_model: PreTrainedModel,
        quantized_models: Dict[str, PreTrainedModel],
        tokenizer: PreTrainedTokenizerFast,
        test_prompts: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compare performance between original and quantized models."""
        results = {}

        # Test original model
        original_times = []
        for prompt in test_prompts:
            time_ms = self.evaluate_inference_time(original_model, tokenizer, prompt)
            original_times.append(time_ms)

        results["original"] = {
            "avg_inference_time_ms": sum(original_times) / len(original_times),
            "total_inference_time_ms": sum(original_times),
        }

        # Test quantized models
        for method, model in quantized_models.items():
            quantized_times = []
            for prompt in test_prompts:
                time_ms = self.evaluate_inference_time(model, tokenizer, prompt)
                quantized_times.append(time_ms)

            avg_time = sum(quantized_times) / len(quantized_times)
            speedup = results["original"]["avg_inference_time_ms"] / avg_time

            results[method] = {
                "avg_inference_time_ms": avg_time,
                "total_inference_time_ms": sum(quantized_times),
                "speedup_ratio": speedup,
            }

        return results

    def get_model_info(self, quantization_method: str) -> Dict[str, any]:
        """Get information about quantized model."""
        model_path = self.quantized_models_dir / quantization_method
        if not model_path.exists():
            return {}

        # Calculate model size
        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return {
            "path": str(model_path),
            "size_mb": total_size / (1024 * 1024),
            "files": [f.name for f in model_path.iterdir() if f.is_file()],
        }
