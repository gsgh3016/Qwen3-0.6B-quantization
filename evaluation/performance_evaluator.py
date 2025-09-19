"""Performance evaluation utilities for quantized models."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import psutil
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast


class PerformanceEvaluator:
    """Evaluates performance metrics of quantized models."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """Initialize with tokenizer."""
        self.tokenizer = tokenizer

    def evaluate_inference_time(
        self,
        model: PreTrainedModel,
        test_prompts: List[str],
        num_runs: int = 3,
        warmup_runs: int = 1,
    ) -> Dict[str, float]:
        """Evaluate inference time for model."""
        # Warmup runs
        for _ in range(warmup_runs):
            for prompt in test_prompts[:1]:  # Use only first prompt for warmup
                try:
                    self._run_inference(model, prompt)
                except:
                    pass

        # Actual timing runs
        times = []
        for _ in range(num_runs):
            run_times = []
            for prompt in test_prompts:
                try:
                    start_time = time.time()
                    self._run_inference(model, prompt)
                    end_time = time.time()
                    run_times.append((end_time - start_time) * 1000)  # Convert to ms
                except Exception as e:
                    print(f"Error timing inference: {e}")
                    continue

            if run_times:
                times.append(sum(run_times) / len(run_times))  # Average time per prompt

        if not times:
            return {"average_time_ms": 0.0, "min_time_ms": 0.0, "max_time_ms": 0.0}

        return {
            "average_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "total_runs": len(times),
        }

    def evaluate_memory_usage(
        self, model: PreTrainedModel, test_prompts: List[str]
    ) -> Dict[str, float]:
        """Evaluate memory usage of model."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run inference to load model into memory
        for prompt in test_prompts[:1]:  # Use first prompt
            try:
                self._run_inference(model, prompt)
                break
            except:
                continue

        # Get memory usage after loading
        loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
        model_memory = loaded_memory - initial_memory

        return {
            "model_memory_mb": model_memory,
            "total_memory_mb": loaded_memory,
            "initial_memory_mb": initial_memory,
        }

    def evaluate_throughput(
        self,
        model: PreTrainedModel,
        test_prompts: List[str],
        duration_seconds: int = 10,
    ) -> Dict[str, float]:
        """Evaluate throughput (prompts per second) of model."""
        start_time = time.time()
        completed_prompts = 0

        while time.time() - start_time < duration_seconds:
            for prompt in test_prompts:
                if time.time() - start_time >= duration_seconds:
                    break

                try:
                    self._run_inference(model, prompt)
                    completed_prompts += 1
                except:
                    continue

        actual_duration = time.time() - start_time
        throughput = completed_prompts / actual_duration if actual_duration > 0 else 0

        return {
            "throughput_prompts_per_second": throughput,
            "completed_prompts": completed_prompts,
            "duration_seconds": actual_duration,
        }

    def compare_models_performance(
        self,
        original_model: PreTrainedModel,
        quantized_models: Dict[str, PreTrainedModel],
        test_prompts: List[str],
    ) -> Dict[str, Dict[str, any]]:
        """Compare performance between original and quantized models."""
        results = {}

        # Evaluate original model
        print("Evaluating original model performance...")
        original_timing = self.evaluate_inference_time(original_model, test_prompts)
        original_memory = self.evaluate_memory_usage(original_model, test_prompts)
        original_throughput = self.evaluate_throughput(original_model, test_prompts)

        results["original"] = {
            **original_timing,
            **original_memory,
            **original_throughput,
        }

        # Evaluate quantized models
        for method, model in quantized_models.items():
            print(f"Evaluating {method} quantized model performance...")

            try:
                quantized_timing = self.evaluate_inference_time(model, test_prompts)
                quantized_memory = self.evaluate_memory_usage(model, test_prompts)
                quantized_throughput = self.evaluate_throughput(model, test_prompts)

                # Calculate speedup ratios
                speedup_ratio = (
                    original_timing["average_time_ms"]
                    / quantized_timing["average_time_ms"]
                    if quantized_timing["average_time_ms"] > 0
                    else 0
                )
                memory_reduction = (
                    (
                        original_memory["model_memory_mb"]
                        - quantized_memory["model_memory_mb"]
                    )
                    / original_memory["model_memory_mb"]
                    if original_memory["model_memory_mb"] > 0
                    else 0
                )
                throughput_improvement = (
                    quantized_throughput["throughput_prompts_per_second"]
                    / original_throughput["throughput_prompts_per_second"]
                    if original_throughput["throughput_prompts_per_second"] > 0
                    else 0
                )

                results[method] = {
                    **quantized_timing,
                    **quantized_memory,
                    **quantized_throughput,
                    "speedup_ratio": speedup_ratio,
                    "memory_reduction_ratio": memory_reduction,
                    "throughput_improvement_ratio": throughput_improvement,
                }

            except Exception as e:
                print(f"Error evaluating {method}: {e}")
                results[method] = {"error": str(e)}

        return results

    def _run_inference(self, model: PreTrainedModel, prompt: str) -> None:
        """Run single inference on model."""
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    def print_performance_summary(self, results: Dict[str, Dict[str, any]]) -> None:
        """Print a summary of performance evaluation results."""
        print("\n=== Performance Summary ===")

        original = results.get("original", {})
        if original:
            print(f"Original Model:")
            print(
                f"  Average inference time: {original.get('average_time_ms', 0):.2f}ms"
            )
            print(f"  Model memory usage: {original.get('model_memory_mb', 0):.2f}MB")
            print(
                f"  Throughput: {original.get('throughput_prompts_per_second', 0):.2f} prompts/sec"
            )

        print(f"\nQuantized Models:")
        for method, perf in results.items():
            if method == "original" or "error" in perf:
                continue

            speedup = perf.get("speedup_ratio", 0)
            memory_reduction = perf.get("memory_reduction_ratio", 0) * 100
            throughput_improvement = perf.get("throughput_improvement_ratio", 0)

            print(f"  {method}:")
            print(f"    Average inference time: {perf.get('average_time_ms', 0):.2f}ms")
            print(f"    Model memory usage: {perf.get('model_memory_mb', 0):.2f}MB")
            print(
                f"    Throughput: {perf.get('throughput_prompts_per_second', 0):.2f} prompts/sec"
            )
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Memory reduction: {memory_reduction:.1f}%")
            print(f"    Throughput improvement: {throughput_improvement:.2f}x")
