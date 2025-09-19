#!/usr/bin/env python3
"""Quick test script for evaluation with progress bars."""

from evaluation.accuracy_evaluator import evaluate_before_quantization
from logs.logger import QuantizationLogger


def main():
    """Run a quick evaluation test."""
    logger = QuantizationLogger()
    logger.info("=" * 60)
    logger.info("QUICK EVALUATION TEST WITH PROGRESS BARS")
    logger.info("=" * 60)

    try:
        # Run only the original model evaluation with progress bars
        logger.info("Running original model evaluation with progress bars...")
        evaluate_before_quantization(logger)

        logger.info("✅ Quick evaluation test completed successfully!")
        logger.info("Progress bars should have been displayed during evaluation.")

    except Exception as e:
        logger.error(f"❌ Quick evaluation test failed: {e}")
        raise


if __name__ == "__main__":
    main()
