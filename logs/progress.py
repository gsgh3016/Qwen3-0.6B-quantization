"""Progress tracking utilities for quantization experiments."""

import time
from typing import List, Optional


class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker."""
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0

    def update(self, increment: int = 1, item: str = "") -> None:
        """Update progress."""
        self.current += increment
        current_time = time.time()

        # Only update display every 0.5 seconds to avoid spam
        if current_time - self.last_update >= 0.5 or self.current >= self.total:
            self._display_progress(item)
            self.last_update = current_time

    def _display_progress(self, item: str = "") -> None:
        """Display current progress."""
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time

        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"

        progress_bar = self._create_progress_bar(percentage)

        if item:
            print(
                f"\r{self.description}: {progress_bar} {percentage:.1f}% ({self.current}/{self.total}) - {item} - {eta_str}",
                end="",
                flush=True,
            )
        else:
            print(
                f"\r{self.description}: {progress_bar} {percentage:.1f}% ({self.current}/{self.total}) - {eta_str}",
                end="",
                flush=True,
            )

        if self.current >= self.total:
            print()  # New line when complete

    def _create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """Create visual progress bar."""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def complete(self) -> None:
        """Mark as complete."""
        self.current = self.total
        self._display_progress()

    def reset(self, total: Optional[int] = None) -> None:
        """Reset progress tracker."""
        if total is not None:
            self.total = total
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0


class BatchProgressTracker:
    """Track progress across multiple batches."""

    def __init__(self, batch_names: List[str]):
        """Initialize batch progress tracker."""
        self.batch_names = batch_names
        self.current_batch = 0
        self.batch_tracker: Optional[ProgressTracker] = None
        self.start_time = time.time()

    def start_batch(self, batch_name: str, total_items: int) -> ProgressTracker:
        """Start tracking a new batch."""
        if batch_name in self.batch_names:
            self.current_batch = self.batch_names.index(batch_name)

        self.batch_tracker = ProgressTracker(
            total_items,
            f"Batch {self.current_batch + 1}/{len(self.batch_names)}: {batch_name}",
        )
        return self.batch_tracker

    def get_overall_progress(self) -> float:
        """Get overall progress across all batches."""
        if self.batch_tracker:
            batch_progress = self.batch_tracker.current / self.batch_tracker.total
            return (self.current_batch + batch_progress) / len(self.batch_names)
        return self.current_batch / len(self.batch_names)

    def get_elapsed_time(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time
