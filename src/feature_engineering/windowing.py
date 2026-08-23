"""Sliding-window utilities."""

from collections.abc import Iterator

import numpy as np


def sliding_windows(
    signal: np.ndarray,
    fs: float,
    window_sec: float,
    step_sec: float,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield complete fixed-length windows.

    Each yielded tuple contains ``(start_sample, end_sample_exclusive, window)``.
    A final incomplete residual is intentionally not padded or returned.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if step_sec <= 0:
        raise ValueError("step_sec must be positive")

    window_samples = int(round(window_sec * fs))
    step_samples = int(round(step_sec * fs))

    if window_samples < 1 or step_samples < 1:
        raise ValueError("window_sec and step_sec must correspond to at least one sample")

    n_samples = len(signal)
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        yield start, end, signal[start:end]


def expected_window_count(
    n_samples: int,
    fs: float,
    window_sec: float,
    step_sec: float,
) -> int:
    """Return floor((N-L)/S)+1 for complete windows, or zero if N < L."""
    window_samples = int(round(window_sec * fs))
    step_samples = int(round(step_sec * fs))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("Window and step must correspond to positive sample counts")
    if n_samples < window_samples:
        return 0
    return 1 + (n_samples - window_samples) // step_samples
