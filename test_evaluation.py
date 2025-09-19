#!/usr/bin/env python3
"""Test script for evaluation functionality."""

from evaluation.dataset_loader import DatasetLoader
from evaluation.evaluation_results import EvaluationResults
from logs.logger import QuantizationLogger


def test_dataset_loading():
    """Test dataset loading functionality."""
    logger = QuantizationLogger()
    logger.info("Testing dataset loading...")

    try:
        dataset_loader = DatasetLoader()

        # Test evaluation dataset loading
        prompts, expected = dataset_loader.load_evaluation_dataset()
        logger.info(f"Loaded {len(prompts)} prompts for evaluation")
        logger.info(f"Sample prompt: {prompts[0] if prompts else 'None'}")

        # Test perplexity dataset loading
        texts = dataset_loader.load_dataset_for_perplexity()
        logger.info(f"Loaded {len(texts)} texts for perplexity evaluation")
        logger.info(f"Sample text: {texts[0][:100] if texts else 'None'}...")

        logger.info("✅ Dataset loading test passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Dataset loading test failed: {e}")
        return False


def test_evaluation_results():
    """Test evaluation results functionality."""
    logger = QuantizationLogger()
    logger.info("Testing evaluation results...")

    try:
        results_manager = EvaluationResults()

        # Test with mock data
        original_results = {
            "total_prompts": 10,
            "successful_predictions": 8,
            "failed_predictions": 2,
        }

        quantized_results = {
            "int8": {
                "accuracy": {
                    "total_prompts": 10,
                    "successful_predictions": 7,
                    "failed_predictions": 3,
                },
                "perplexity": {"perplexity": 15.5},
            }
        }

        # Test saving results
        results_file = results_manager.save_evaluation_results(
            original_results=original_results,
            quantized_results=quantized_results,
            dataset_name="test_dataset",
            evaluation_type="test",
        )

        logger.info(f"Results saved to: {results_file}")

        # Test loading results
        loaded_results = results_manager.load_latest_results("test")
        logger.info(f"Loaded results: {bool(loaded_results)}")

        logger.info("✅ Evaluation results test passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Evaluation results test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger = QuantizationLogger()
    logger.info("=" * 60)
    logger.info("EVALUATION FUNCTIONALITY TEST")
    logger.info("=" * 60)

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Evaluation Results", test_evaluation_results),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        if test_func():
            passed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")
    logger.info("=" * 60)

    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("❌ Some tests failed!")


if __name__ == "__main__":
    main()
