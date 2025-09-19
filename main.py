"""Main entry point for Qwen3-0.6B quantization experiments."""

from evaluation.accuracy_evaluator import (
    evaluate_after_quantization,
    evaluate_before_quantization,
)
from experiments.quantization_experiments import run_quantization_experiment
from logs.logger import QuantizationLogger
from schemas.schema_builder import load_quantization_experiment_config


def main() -> None:
    """Run complete quantization experiment pipeline."""
    logger = QuantizationLogger()
    config = load_quantization_experiment_config()

    try:
        logger.info("=" * 80)
        logger.info("QWEN3-0.6B QUANTIZATION EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Dataset: {config.evaluation.dataset}")
        logger.info(
            f"Quantization methods: {[m.name for m in config.quantization_methods]}"
        )
        logger.info(f"Metrics: {config.evaluation.metrics}")
        logger.info("=" * 80)

        # 1. Evaluate original model before quantization
        logger.info("\n🔍 STEP 1: Evaluating original model...")
        evaluate_before_quantization(logger)

        # 2. Run quantization experiments
        logger.info("\n⚙️  STEP 2: Running quantization experiments...")
        quantize(logger)

        # 3. Evaluate quantized models after quantization
        logger.info("\n📊 STEP 3: Evaluating quantized models...")
        evaluate_after_quantization(logger)

        logger.info("\n✅ Experiment completed successfully!")
        logger.info("Check the results directory for detailed evaluation results.")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


def quantize(logger: QuantizationLogger) -> None:
    """Run quantization experiments."""
    logger.info("Starting quantization experiments...")
    results = run_quantization_experiment(logger)

    if results["success"]:
        logger.info("Quantization experiments completed successfully!")
        logger.info(f"Generated {len(results['quantized_paths'])} quantized models")
    else:
        logger.error("Quantization experiments failed!")


if __name__ == "__main__":
    main()
