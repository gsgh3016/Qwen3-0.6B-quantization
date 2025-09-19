"""Main quantization experiment script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from logs.logger import QuantizationLogger
from quantization.quantizer import Quantizer
from schemas.schema_builder import load_quantization_experiment_config


def run_quantization_experiment(logger: QuantizationLogger = None) -> Dict[str, any]:
    """Run complete quantization experiment."""
    if logger is None:
        logger = QuantizationLogger()

    logger.experiment_start("Qwen3-0.6B Quantization Experiment")

    # Load configuration
    config = load_quantization_experiment_config()

    # Get original model name and cache directory
    original_model_name = config.model.original.name
    original_model_cache_dir = config.model.original.cache_dir

    # Initialize quantizer
    quantizer = Quantizer(
        model_path=original_model_name,
        output_dir=config.model.quantized.cache_dir,
        logger=logger,
    )

    # Run quantization for each method
    quantized_paths = {}
    quantization_methods = config.quantization_methods

    logger.info(f"Found {len(quantization_methods)} quantization methods to test:")
    for method in quantization_methods:
        logger.info(f"  - {method.name} ({method.bits}-bit, {method.method})")

    for method_config in quantization_methods:
        method_name = method_config.name

        try:
            if method_name == "int8":
                quantized_path = quantizer.quantize_int8_dynamic()
            elif method_name == "int4":
                quantized_path = quantizer.quantize_int4_static()
            elif method_name == "dynamic":
                quantized_path = quantizer.quantize_dynamic()
            else:
                logger.warning(f"Unknown quantization method: {method_name}")
                continue

            quantized_paths[method_name] = quantized_path

        except Exception as e:
            logger.quantization_error(method_name, str(e))
            continue

    # Compare model sizes
    size_comparison = quantizer.compare_model_sizes(
        original_model_cache_dir, quantized_paths
    )
    logger.model_size_comparison(size_comparison)

    # Save experiment results
    if config.experiment.save_results:
        results_dir = Path(config.experiment.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Convert quantization methods to serializable format
        quantization_methods_serializable = [
            {"name": method.name, "bits": method.bits, "method": method.method}
            for method in quantization_methods
        ]

        results = {
            "quantization_methods": quantization_methods_serializable,
            "quantized_paths": quantized_paths,
            "size_comparison": size_comparison,
            "original_model_name": original_model_name,
            "original_model_cache_dir": original_model_cache_dir,
        }

        results_file = results_dir / "quantization_experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Experiment results saved to: {results_file}")

    logger.experiment_end("Qwen3-0.6B Quantization Experiment")
    return {
        "quantized_paths": quantized_paths,
        "size_comparison": size_comparison,
        "original_model_name": original_model_name,
        "original_model_cache_dir": original_model_cache_dir,
        "success": len(quantized_paths) > 0,
    }


if __name__ == "__main__":
    run_quantization_experiment()
