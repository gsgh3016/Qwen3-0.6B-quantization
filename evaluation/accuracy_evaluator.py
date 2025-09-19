"""Accuracy evaluation utilities for quantized models."""

from __future__ import annotations

from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from core.model_manager import load_original_model, load_original_tokenizer
from evaluation.dataset_loader import DatasetLoader
from evaluation.evaluation_results import EvaluationResults
from logs.logger import QuantizationLogger
from schemas import QuantizationComparisonResult


class AccuracyEvaluator:
    """Evaluates accuracy of quantized models compared to original."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """Initialize with tokenizer."""
        self.tokenizer = tokenizer

    def evaluate_model_accuracy(
        self,
        model: PreTrainedModel,
        test_prompts: List[str],
        quantization_method: str = None,
    ) -> Dict[str, any]:
        """Evaluate accuracy of a single model."""
        results = {
            "quantization_method": quantization_method,
            "total_prompts": len(test_prompts),
            "successful_predictions": 0,
            "failed_predictions": 0,
            "predictions": [],
        }

        # Create progress bar for accuracy evaluation
        progress_bar = tqdm(
            enumerate(test_prompts),
            total=len(test_prompts),
            desc=f"Evaluating {quantization_method or 'model'} accuracy",
            unit="prompt",
        )

        for i, prompt in progress_bar:
            try:
                # Get model prediction
                encoded = self.tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)

                with torch.no_grad():
                    logits = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits[0, -1, :]

                probabilities = torch.softmax(logits, dim=-1)
                predicted_token_id = torch.argmax(logits).item()
                predicted_token = self.tokenizer.decode(
                    predicted_token_id, skip_special_tokens=True
                )
                confidence = probabilities[predicted_token_id].item()

                results["predictions"].append(
                    {
                        "prompt": prompt,
                        "predicted_token": predicted_token,
                        "token_id": predicted_token_id,
                        "confidence": confidence,
                        "success": True,
                    }
                )
                results["successful_predictions"] += 1

                # Update progress bar with current success rate
                success_rate = results["successful_predictions"] / (i + 1)
                progress_bar.set_postfix(
                    {
                        "success_rate": f"{success_rate:.3f}",
                        "successful": results["successful_predictions"],
                        "failed": results["failed_predictions"],
                    }
                )

            except Exception as e:
                results["predictions"].append(
                    {"prompt": prompt, "error": str(e), "success": False}
                )
                results["failed_predictions"] += 1

                # Update progress bar with current success rate
                success_rate = results["successful_predictions"] / (i + 1)
                progress_bar.set_postfix(
                    {
                        "success_rate": f"{success_rate:.3f}",
                        "successful": results["successful_predictions"],
                        "failed": results["failed_predictions"],
                    }
                )

        return results

    def compare_models_accuracy(
        self,
        original_model: PreTrainedModel,
        quantized_models: Dict[str, PreTrainedModel],
        test_prompts: List[str],
    ) -> Dict[str, QuantizationComparisonResult]:
        """Compare accuracy between original and quantized models."""
        # Evaluate original model
        original_results = self.evaluate_model_accuracy(
            original_model, test_prompts, "original"
        )

        comparison_results = {}

        for method, quantized_model in quantized_models.items():
            # Evaluate quantized model
            quantized_results = self.evaluate_model_accuracy(
                quantized_model, test_prompts, method
            )

            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(
                original_results, quantized_results
            )

            # Create comparison result
            comparison_result = QuantizationComparisonResult(
                original=original_results,
                quantized=quantized_results,
                accuracy_drop=accuracy_metrics["accuracy_drop"],
                size_reduction_ratio=accuracy_metrics.get("size_reduction_ratio"),
                speedup_ratio=accuracy_metrics.get("speedup_ratio"),
            )

            comparison_results[method] = comparison_result

        return comparison_results

    def _calculate_accuracy_metrics(
        self, original_results: Dict[str, any], quantized_results: Dict[str, any]
    ) -> Dict[str, float]:
        """Calculate accuracy comparison metrics."""
        total_prompts = original_results["total_prompts"]
        original_successful = original_results["successful_predictions"]
        quantized_successful = quantized_results["successful_predictions"]

        # Calculate accuracy rates
        original_accuracy = (
            original_successful / total_prompts if total_prompts > 0 else 0
        )
        quantized_accuracy = (
            quantized_successful / total_prompts if total_prompts > 0 else 0
        )

        # Calculate accuracy drop
        accuracy_drop = original_accuracy - quantized_accuracy

        # Calculate token-level accuracy (exact matches)
        token_matches = 0
        total_comparable = 0

        for orig_pred, quant_pred in zip(
            original_results["predictions"], quantized_results["predictions"]
        ):
            if orig_pred["success"] and quant_pred["success"]:
                total_comparable += 1
                if orig_pred["token_id"] == quant_pred["token_id"]:
                    token_matches += 1

        token_accuracy = token_matches / total_comparable if total_comparable > 0 else 0
        token_accuracy_drop = (
            1.0 - token_accuracy
        )  # Original accuracy is 1.0 for exact matches

        return {
            "accuracy_drop": accuracy_drop,
            "token_accuracy_drop": token_accuracy_drop,
            "original_accuracy": original_accuracy,
            "quantized_accuracy": quantized_accuracy,
            "token_accuracy": token_accuracy,
        }

    def evaluate_perplexity(
        self,
        model: PreTrainedModel,
        test_texts: List[str],
        quantization_method: str = None,
    ) -> Dict[str, float]:
        """Evaluate perplexity of model on test texts."""
        total_loss = 0.0
        total_tokens = 0
        processed_texts = 0

        # Create progress bar for perplexity evaluation
        progress_bar = tqdm(
            test_texts,
            desc=f"Evaluating {quantization_method or 'model'} perplexity",
            unit="text",
        )

        for text in progress_bar:
            try:
                encoded = self.tokenizer(text, return_tensors="pt")
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    loss = outputs.loss
                    total_loss += loss.item() * input_ids.size(1)
                    total_tokens += input_ids.size(1)
                    processed_texts += 1

                # Update progress bar with current perplexity
                if total_tokens > 0:
                    avg_loss = total_loss / total_tokens
                    current_perplexity = torch.exp(torch.tensor(avg_loss)).item()
                    progress_bar.set_postfix(
                        {
                            "perplexity": f"{current_perplexity:.2f}",
                            "processed": processed_texts,
                            "tokens": total_tokens,
                        }
                    )

            except Exception as e:
                print(f"Error evaluating perplexity for text: {e}")
                continue

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")

        # Calculate perplexity with safety checks
        if avg_loss == float("inf") or avg_loss > 20:  # Reasonable upper bound
            perplexity = float("inf")
        else:
            try:
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                # Cap perplexity at a reasonable value
                if perplexity > 1e6:
                    perplexity = float("inf")
            except (OverflowError, RuntimeError):
                perplexity = float("inf")

        return {
            "perplexity": perplexity,
            "average_loss": avg_loss,
            "total_tokens": total_tokens,
            "quantization_method": quantization_method,
        }


def evaluate_before_quantization(logger: QuantizationLogger) -> None:
    """Evaluate original model before quantization."""
    logger.info("Evaluating original model before quantization...")

    try:
        # Load original model and tokenizer
        model = load_original_model()
        tokenizer = load_original_tokenizer()

        # Create evaluator and dataset loader
        evaluator = AccuracyEvaluator(tokenizer)
        dataset_loader = DatasetLoader()

        # Load evaluation dataset from config
        test_prompts, expected_outputs = dataset_loader.load_evaluation_dataset()

        # Evaluate original model
        results = evaluator.evaluate_model_accuracy(
            model=model, test_prompts=test_prompts, quantization_method="original"
        )

        # Evaluate perplexity
        texts = dataset_loader.load_dataset_for_perplexity()
        perplexity_results = evaluator.evaluate_perplexity(
            model=model, test_texts=texts, quantization_method="original"
        )

        # Save results
        results_manager = EvaluationResults()
        dataset_name = dataset_loader.config.evaluation.dataset

        # Prepare results for saving
        original_results = {"accuracy": results, "perplexity": perplexity_results}

        # Save original model results
        results_file = results_manager.save_evaluation_results(
            original_results=original_results,
            quantized_results={},
            dataset_name=dataset_name,
            evaluation_type="original",
        )

        # Print summary
        results_manager.print_evaluation_summary(
            original_results=results, quantized_results={}, dataset_name=dataset_name
        )

        logger.info("Original model evaluation completed:")
        logger.info(f"  - Total prompts: {results['total_prompts']}")
        logger.info(f"  - Successful predictions: {results['successful_predictions']}")
        logger.info(f"  - Failed predictions: {results['failed_predictions']}")
        logger.info(f"  - Perplexity: {perplexity_results['perplexity']:.4f}")
        logger.info(f"  - Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Original model evaluation failed: {e}")
        raise


def evaluate_after_quantization(logger: QuantizationLogger) -> None:
    """Evaluate quantized models after quantization."""
    logger.info("Evaluating quantized models after quantization...")

    try:
        # Load original model and tokenizer
        model = load_original_model()
        tokenizer = load_original_tokenizer()

        # Create evaluator and dataset loader
        evaluator = AccuracyEvaluator(tokenizer)
        dataset_loader = DatasetLoader()

        # Load evaluation dataset from config
        test_prompts, expected_outputs = dataset_loader.load_evaluation_dataset()
        texts = dataset_loader.load_dataset_for_perplexity()

        # Evaluate original model for comparison
        original_results = evaluator.evaluate_model_accuracy(
            model=model, test_prompts=test_prompts, quantization_method="original"
        )
        original_perplexity = evaluator.evaluate_perplexity(
            model=model, test_texts=texts, quantization_method="original"
        )

        logger.info("Original model results:")
        logger.info(
            f"  - Successful predictions: {original_results['successful_predictions']}"
        )
        logger.info(f"  - Perplexity: {original_perplexity['perplexity']:.4f}")

        # Load and evaluate quantized models
        from core.model_manager import load_quantized_model
        from schemas.schema_builder import load_quantization_experiment_config

        config = load_quantization_experiment_config()
        quantized_models = {}
        quantized_results = {}

        for method in config.quantization_methods:
            try:
                quantized_model = load_quantized_model(method.name)
                quantized_models[method.name] = quantized_model

                # Evaluate quantized model
                accuracy_results = evaluator.evaluate_model_accuracy(
                    model=quantized_model,
                    test_prompts=test_prompts,
                    quantization_method=method.name,
                )
                perplexity_results = evaluator.evaluate_perplexity(
                    model=quantized_model,
                    test_texts=texts,
                    quantization_method=method.name,
                )

                # Store results
                quantized_results[method.name] = {
                    "accuracy": accuracy_results,
                    "perplexity": perplexity_results,
                }

                logger.info(f"Quantized model ({method.name}) results:")
                logger.info(
                    f"  - Successful predictions: {accuracy_results['successful_predictions']}"
                )
                logger.info(f"  - Perplexity: {perplexity_results['perplexity']:.4f}")

            except Exception as e:
                logger.warning(
                    f"Failed to load/evaluate quantized model {method.name}: {e}"
                )
                continue

        # Save and display comprehensive results
        results_manager = EvaluationResults()
        dataset_name = dataset_loader.config.evaluation.dataset

        # Prepare original results for comparison
        original_results_for_save = {
            "accuracy": original_results,
            "perplexity": original_perplexity,
        }

        # Save comprehensive results
        results_file = results_manager.save_evaluation_results(
            original_results=original_results_for_save,
            quantized_results=quantized_results,
            dataset_name=dataset_name,
            evaluation_type="comprehensive",
        )

        # Print comprehensive summary
        results_manager.print_evaluation_summary(
            original_results=original_results,
            quantized_results=quantized_results,
            dataset_name=dataset_name,
        )

        logger.info("Quantized model evaluation completed")
        logger.info(f"  - Comprehensive results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Quantized model evaluation failed: {e}")
        raise
