"""Accuracy comparison between original and quantized models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from core.model_manager import load_original_model, load_original_tokenizer
from evaluation.quantization_evaluation import evaluate_quantized_model
from logs.logger import QuantizationLogger
from schemas.schema_builder import load_quantization_experiment_config


def compare_accuracy(
    test_prompts: List[str] = None, quantization_methods: List[str] = None
) -> Dict[str, any]:
    """Compare accuracy between original and quantized models."""
    print("=== Accuracy Comparison Experiment ===")

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

    # Default quantization methods if not provided
    if quantization_methods is None:
        quantization_methods = [
            method["name"] for method in config.quantization_methods
        ]

    print(
        f"Testing {len(test_prompts)} prompts with {len(quantization_methods)} quantization methods"
    )

    # Get original model and tokenizer
    original_model = get_original_model()
    tokenizer = get_original_tokenizer()

    results = {
        "test_prompts": test_prompts,
        "quantization_methods": quantization_methods,
        "original_results": {},
        "quantized_results": {},
        "accuracy_comparison": {},
    }

    # Test original model
    print("\n--- Testing Original Model ---")
    original_results = {}

    for i, prompt in enumerate(test_prompts, 1):
        print(f"Testing prompt {i}/{len(test_prompts)}: {prompt}")

        try:
            # Get original model prediction
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(original_model.device)
            attention_mask = encoded["attention_mask"].to(original_model.device)

            with torch.no_grad():
                logits = original_model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[0, -1, :]

            probabilities = torch.softmax(logits, dim=-1)
            predicted_token_id = torch.argmax(logits).item()
            predicted_token = tokenizer.decode(
                predicted_token_id, skip_special_tokens=True
            )
            confidence = probabilities[predicted_token_id].item()

            original_results[prompt] = {
                "predicted_token": predicted_token,
                "confidence": confidence,
                "token_id": predicted_token_id,
            }

        except Exception as e:
            print(f"Error testing original model with prompt '{prompt}': {e}")
            original_results[prompt] = {"error": str(e)}

    results["original_results"] = original_results

    # Test quantized models
    for method in quantization_methods:
        print(f"\n--- Testing {method} Quantized Model ---")
        quantized_results = {}

        for i, prompt in enumerate(test_prompts, 1):
            print(f"Testing prompt {i}/{len(test_prompts)}: {prompt}")

            try:
                # Get quantized model prediction
                evaluation_result = evaluate_quantized_model(
                    prompt=prompt, quantization_method=method, display=False
                )

                greedy_prediction = evaluation_result.greedy
                quantized_results[prompt] = {
                    "predicted_token": greedy_prediction.token_text,
                    "confidence": greedy_prediction.probability,
                    "token_id": greedy_prediction.token_id,
                }

            except Exception as e:
                print(f"Error testing {method} model with prompt '{prompt}': {e}")
                quantized_results[prompt] = {"error": str(e)}

        results["quantized_results"][method] = quantized_results

        # Compare accuracy
        accuracy_comparison = compare_predictions(original_results, quantized_results)
        results["accuracy_comparison"][method] = accuracy_comparison

        print(f"{method} accuracy: {accuracy_comparison['accuracy']:.2%}")

    # Save results
    if config.experiment.save_results:
        results_dir = Path(config.experiment.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "accuracy_comparison_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nAccuracy comparison results saved to: {results_file}")

    print("\n=== Accuracy Comparison Completed ===")
    return results


def compare_predictions(
    original_results: Dict[str, any], quantized_results: Dict[str, any]
) -> Dict[str, any]:
    """Compare predictions between original and quantized models."""
    total_prompts = len(original_results)
    correct_predictions = 0
    confidence_differences = []

    for prompt in original_results:
        if prompt not in quantized_results:
            continue

        orig_result = original_results[prompt]
        quant_result = quantized_results[prompt]

        # Skip if either result has an error
        if "error" in orig_result or "error" in quant_result:
            continue

        # Check if predictions match
        if orig_result["token_id"] == quant_result["token_id"]:
            correct_predictions += 1

        # Calculate confidence difference
        orig_conf = orig_result.get("confidence", 0)
        quant_conf = quant_result.get("confidence", 0)
        confidence_diff = abs(orig_conf - quant_conf)
        confidence_differences.append(confidence_diff)

    accuracy = correct_predictions / total_prompts if total_prompts > 0 else 0
    avg_confidence_diff = (
        sum(confidence_differences) / len(confidence_differences)
        if confidence_differences
        else 0
    )

    return {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_prompts": total_prompts,
        "average_confidence_difference": avg_confidence_diff,
    }


if __name__ == "__main__":
    compare_accuracy()
