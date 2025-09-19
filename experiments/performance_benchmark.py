"""Performance benchmark for quantized models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from core.model_manager import load_original_model, load_original_tokenizer
from logs.logger import QuantizationLogger
from quantization.quantized_model import QuantizedModelManager
from schemas.schema_builder import load_quantization_experiment_config


def benchmark_performance(
    test_prompts: List[str] = None, num_runs: int = 5
) -> Dict[str, any]:
    """Benchmark performance of original and quantized models."""
    print("=== Performance Benchmark Experiment ===")

    # Load configuration
    config = load_quantization_experiment_config()

    # Default test prompts if not provided
    if test_prompts is None:
        test_prompts = [
            "1+1=",
            "The capital of France is",
            "What is the meaning of life?",
            "Translate to French: Hello world",
            "Explain quantum computing in simple terms",
        ]

    print(f"Benchmarking with {len(test_prompts)} prompts, {num_runs} runs each")

    # Get original model and tokenizer
    original_model = get_original_model()
    tokenizer = get_original_tokenizer()

    # Initialize quantized model manager
    quantized_manager = QuantizedModelManager(config.model["quantized"]["cache_dir"])
    available_methods = quantized_manager.list_available_quantizations()

    print(f"Available quantization methods: {available_methods}")

    results = {
        "test_prompts": test_prompts,
        "num_runs": num_runs,
        "original_performance": {},
        "quantized_performance": {},
        "performance_comparison": {},
    }

    # Benchmark original model
    print("\n--- Benchmarking Original Model ---")
    original_times = []
    original_memory_usage = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"Benchmarking prompt {i}/{len(test_prompts)}: {prompt}")

        prompt_times = []
        for run in range(num_runs):
            try:
                encoded = tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(original_model.device)
                attention_mask = encoded["attention_mask"].to(original_model.device)

                start_time = time.time()
                with torch.no_grad():
                    _ = original_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                end_time = time.time()

                inference_time = (end_time - start_time) * 1000  # Convert to ms
                prompt_times.append(inference_time)

            except Exception as e:
                print(f"Error in run {run + 1}: {e}")
                continue

        if prompt_times:
            avg_time = sum(prompt_times) / len(prompt_times)
            original_times.append(avg_time)
            print(f"  Average time: {avg_time:.2f}ms")

    results["original_performance"] = {
        "average_inference_time_ms": sum(original_times) / len(original_times)
        if original_times
        else 0,
        "total_inference_time_ms": sum(original_times),
        "individual_times": original_times,
    }

    # Benchmark quantized models
    for method in available_methods:
        print(f"\n--- Benchmarking {method} Quantized Model ---")

        try:
            quantized_model = quantized_manager.load_quantized_model(method)
            quantized_times = []

            for i, prompt in enumerate(test_prompts, 1):
                print(f"Benchmarking prompt {i}/{len(test_prompts)}: {prompt}")

                prompt_times = []
                for run in range(num_runs):
                    try:
                        encoded = tokenizer(prompt, return_tensors="pt")
                        input_ids = encoded["input_ids"].to(quantized_model.device)
                        attention_mask = encoded["attention_mask"].to(
                            quantized_model.device
                        )

                        start_time = time.time()
                        with torch.no_grad():
                            _ = quantized_model(
                                input_ids=input_ids, attention_mask=attention_mask
                            )
                        end_time = time.time()

                        inference_time = (end_time - start_time) * 1000  # Convert to ms
                        prompt_times.append(inference_time)

                    except Exception as e:
                        print(f"Error in run {run + 1}: {e}")
                        continue

                if prompt_times:
                    avg_time = sum(prompt_times) / len(prompt_times)
                    quantized_times.append(avg_time)
                    print(f"  Average time: {avg_time:.2f}ms")

            avg_inference_time = (
                sum(quantized_times) / len(quantized_times) if quantized_times else 0
            )
            original_avg_time = results["original_performance"][
                "average_inference_time_ms"
            ]
            speedup = (
                original_avg_time / avg_inference_time if avg_inference_time > 0 else 0
            )

            results["quantized_performance"][method] = {
                "average_inference_time_ms": avg_inference_time,
                "total_inference_time_ms": sum(quantized_times),
                "individual_times": quantized_times,
                "speedup_ratio": speedup,
            }

            results["performance_comparison"][method] = {
                "speedup_ratio": speedup,
                "time_reduction_ms": original_avg_time - avg_inference_time,
                "time_reduction_percent": (
                    (original_avg_time - avg_inference_time) / original_avg_time
                )
                * 100
                if original_avg_time > 0
                else 0,
            }

            print(
                f"{method} average time: {avg_inference_time:.2f}ms (speedup: {speedup:.2f}x)"
            )

        except Exception as e:
            print(f"Error benchmarking {method}: {e}")
            results["quantized_performance"][method] = {"error": str(e)}

    # Save results
    if config.experiment.save_results:
        results_dir = Path(config.experiment.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "performance_benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nPerformance benchmark results saved to: {results_file}")

    print("\n=== Performance Benchmark Completed ===")
    return results


def print_performance_summary(results: Dict[str, any]) -> None:
    """Print a summary of performance benchmark results."""
    print("\n=== Performance Summary ===")

    original_perf = results["original_performance"]
    print(f"Original Model:")
    print(
        f"  Average inference time: {original_perf['average_inference_time_ms']:.2f}ms"
    )

    print(f"\nQuantized Models:")
    for method, perf in results["quantized_performance"].items():
        if "error" in perf:
            print(f"  {method}: Error - {perf['error']}")
            continue

        speedup = perf["speedup_ratio"]
        time_reduction = results["performance_comparison"][method][
            "time_reduction_percent"
        ]

        print(f"  {method}:")
        print(f"    Average inference time: {perf['average_inference_time_ms']:.2f}ms")
        print(f"    Speedup: {speedup:.2f}x")
        print(f"    Time reduction: {time_reduction:.1f}%")


if __name__ == "__main__":
    results = benchmark_performance()
    print_performance_summary(results)
