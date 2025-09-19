"""Centralized logging for quantization experiments."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class QuantizationLogger:
    """Centralized logger for quantization experiments."""

    def __init__(
        self, name: str = "quantization", log_dir: str = "logs", max_log_files: int = 10
    ):
        """Initialize logger with name and log directory."""
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.max_log_files = max_log_files

        # Clean up old log files
        self._cleanup_old_logs()

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def _cleanup_old_logs(self) -> None:
        """Remove old log files if they exceed max_log_files limit."""
        log_files = list(self.log_dir.glob(f"{self.name}_*.log"))

        if len(log_files) >= self.max_log_files:
            # Sort by modification time (oldest first)
            log_files.sort(key=lambda x: x.stat().st_mtime)

            # Remove oldest files
            files_to_remove = log_files[: -self.max_log_files + 1]
            for log_file in files_to_remove:
                try:
                    log_file.unlink()
                    print(f"Removed old log file: {log_file.name}")
                except OSError as e:
                    print(f"Failed to remove log file {log_file.name}: {e}")

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def experiment_start(self, experiment_name: str) -> None:
        """Log experiment start."""
        self.info(f"=== Starting {experiment_name} ===")

    def experiment_end(self, experiment_name: str) -> None:
        """Log experiment end."""
        self.info(f"=== {experiment_name} Completed ===")

    def quantization_start(self, method: str) -> None:
        """Log quantization start."""
        self.info(f"--- Quantizing with {method} ---")

    def quantization_success(self, method: str, output_path: str) -> None:
        """Log quantization success."""
        self.info(f"✓ {method} quantization completed -> {output_path}")

    def quantization_error(self, method: str, error: str) -> None:
        """Log quantization error."""
        self.error(f"✗ Error in {method} quantization: {error}")

    def evaluation_start(self, model_type: str, method: Optional[str] = None) -> None:
        """Log evaluation start."""
        if method:
            self.info(f"--- Testing {method} {model_type} Model ---")
        else:
            self.info(f"--- Testing {model_type} Model ---")

    def evaluation_progress(self, current: int, total: int, item: str) -> None:
        """Log evaluation progress."""
        self.info(f"Testing {item} ({current}/{total})")

    def evaluation_error(self, item: str, error: str) -> None:
        """Log evaluation error."""
        self.error(f"Error testing {item}: {error}")

    def performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"{key}: {value:.4f}")
            else:
                self.info(f"{key}: {value}")

    def model_size_comparison(self, comparison: Dict[str, float]) -> None:
        """Log model size comparison."""
        self.info("--- Model Size Comparison ---")
        original_size = comparison.get("original_mb", 0)
        self.info(f"Original: {original_size:.2f} MB")

        for key, value in comparison.items():
            if key.endswith("_mb") and key != "original_mb":
                method = key.replace("_mb", "")
                reduction = comparison.get(f"{method}_reduction_ratio", 0)
                self.info(f"{method}: {value:.2f} MB (reduction: {reduction:.1%})")


def setup_logging(
    name: str = "quantization", log_dir: str = "logs"
) -> QuantizationLogger:
    """Setup and return a configured logger."""
    return QuantizationLogger(name, log_dir)
