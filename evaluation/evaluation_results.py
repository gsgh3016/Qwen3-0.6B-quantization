"""Evaluation results management and reporting."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from logs.logger import QuantizationLogger
from schemas.schema_builder import load_quantization_experiment_config


class EvaluationResults:
    """Manages evaluation results storage and reporting."""

    def __init__(self):
        """Initialize evaluation results manager."""
        self.config = load_quantization_experiment_config()
        self.logger = QuantizationLogger()
        self.results_dir = Path(self.config.experiment.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_evaluation_results(
        self,
        original_results: Dict[str, Any],
        quantized_results: Dict[str, Dict[str, Any]],
        dataset_name: str,
        evaluation_type: str = "accuracy",
    ) -> str:
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{evaluation_type}_evaluation_{timestamp}.json"
        filepath = self.results_dir / filename

        # Prepare results data
        results_data = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "evaluation_type": evaluation_type,
            "original_model": original_results,
            "quantized_models": quantized_results,
            "summary": self._generate_summary(original_results, quantized_results),
        }

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Evaluation results saved to: {filepath}")
        return str(filepath)

    def _generate_summary(
        self,
        original_results: Dict[str, Any],
        quantized_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate summary of evaluation results."""
        accuracy_payload = self._extract_accuracy_payload(original_results)
        summary = {
            "total_prompts": accuracy_payload.get("total_prompts", 0),
            "original_accuracy": self._calculate_accuracy_rate(original_results),
            "quantized_models": {},
        }

        for method, results in quantized_results.items():
            accuracy_results = self._extract_accuracy_payload(results)
            perplexity_results = self._extract_perplexity_payload(results)

            summary["quantized_models"][method] = {
                "accuracy": self._calculate_accuracy_rate(accuracy_results),
                "perplexity": perplexity_results.get("perplexity", 0.0),
                "accuracy_drop": summary["original_accuracy"]
                - self._calculate_accuracy_rate(accuracy_results),
            }

        return summary

    def _calculate_accuracy_rate(self, results: Dict[str, Any]) -> float:
        """Calculate accuracy rate from results."""
        payload = self._extract_accuracy_payload(results)
        total = payload.get("total_prompts", 0)
        successful = payload.get("successful_predictions", 0)
        return successful / total if total > 0 else 0.0

    @staticmethod
    def _extract_accuracy_payload(results: Dict[str, Any]) -> Dict[str, Any]:
        """Return the accuracy payload regardless of wrapper structure."""
        if isinstance(results, dict) and "accuracy" in results:
            nested = results["accuracy"]
            if isinstance(nested, dict):
                return nested
        return results

    @staticmethod
    def _extract_perplexity_payload(results: Dict[str, Any]) -> Dict[str, Any]:
        """Return the perplexity payload if present."""
        if isinstance(results, dict) and "perplexity" in results:
            nested = results["perplexity"]
            if isinstance(nested, dict):
                return nested
        return results

    def print_evaluation_summary(
        self,
        original_results: Dict[str, Any],
        quantized_results: Dict[str, Dict[str, Any]],
        dataset_name: str,
    ) -> None:
        """Print formatted evaluation summary."""
        print("\n" + "=" * 80)
        print(f"EVALUATION RESULTS - Dataset: {dataset_name}")
        print("=" * 80)

        # Original model results
        print(f"\n📊 ORIGINAL MODEL:")
        print(f"  Total prompts: {original_results.get('total_prompts', 0)}")
        print(
            f"  Successful predictions: {original_results.get('successful_predictions', 0)}"
        )
        print(f"  Failed predictions: {original_results.get('failed_predictions', 0)}")
        print(f"  Accuracy rate: {self._calculate_accuracy_rate(original_results):.4f}")

        # Quantized models results
        print(f"\n🔧 QUANTIZED MODELS:")
        for method, results in quantized_results.items():
            if "accuracy" in results:
                accuracy_results = results["accuracy"]
                perplexity_results = results.get("perplexity", {})
            else:
                accuracy_results = results
                perplexity_results = results

            accuracy_rate = self._calculate_accuracy_rate(accuracy_results)
            perplexity = perplexity_results.get("perplexity", 0.0)
            accuracy_drop = (
                self._calculate_accuracy_rate(original_results) - accuracy_rate
            )

            print(f"\n  {method.upper()}:")
            print(f"    Accuracy rate: {accuracy_rate:.4f}")
            if perplexity == float("inf"):
                print(f"    Perplexity: ∞ (infinite)")
            else:
                print(f"    Perplexity: {perplexity:.4f}")
            print(f"    Accuracy drop: {accuracy_drop:.4f}")

        print("\n" + "=" * 80)

    def load_latest_results(self, evaluation_type: str = "accuracy") -> Dict[str, Any]:
        """Load the most recent evaluation results."""
        pattern = f"{evaluation_type}_evaluation_*.json"
        result_files = list(self.results_dir.glob(pattern))

        if not result_files:
            self.logger.warning(f"No {evaluation_type} evaluation results found")
            return {}

        # Get the most recent file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare_evaluations(self, file1: str, file2: str) -> None:
        """Compare two evaluation results."""
        try:
            with open(file1, "r", encoding="utf-8") as f:
                results1 = json.load(f)
            with open(file2, "r", encoding="utf-8") as f:
                results2 = json.load(f)

            print("\n" + "=" * 80)
            print("EVALUATION COMPARISON")
            print("=" * 80)
            print(f"File 1: {Path(file1).name}")
            print(f"File 2: {Path(file2).name}")
            print("=" * 80)

            # Compare original models
            orig1 = results1.get("original_model", {})
            orig2 = results2.get("original_model", {})

            print(f"\n📊 ORIGINAL MODEL COMPARISON:")
            print(f"  Accuracy 1: {self._calculate_accuracy_rate(orig1):.4f}")
            print(f"  Accuracy 2: {self._calculate_accuracy_rate(orig2):.4f}")
            print(
                f"  Difference: {self._calculate_accuracy_rate(orig2) - self._calculate_accuracy_rate(orig1):.4f}"
            )

            # Compare quantized models
            quant1 = results1.get("quantized_models", {})
            quant2 = results2.get("quantized_models", {})

            print(f"\n🔧 QUANTIZED MODELS COMPARISON:")
            for method in set(quant1.keys()) | set(quant2.keys()):
                if method in quant1 and method in quant2:
                    acc1 = self._calculate_accuracy_rate(quant1[method])
                    acc2 = self._calculate_accuracy_rate(quant2[method])
                    print(
                        f"  {method}: {acc1:.4f} → {acc2:.4f} (Δ: {acc2 - acc1:+.4f})"
                    )
                elif method in quant1:
                    print(
                        f"  {method}: {self._calculate_accuracy_rate(quant1[method]):.4f} → N/A"
                    )
                else:
                    print(
                        f"  {method}: N/A → {self._calculate_accuracy_rate(quant2[method]):.4f}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to compare evaluations: {e}")
            raise
