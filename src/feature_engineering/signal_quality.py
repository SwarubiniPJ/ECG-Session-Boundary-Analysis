"""RR-interval quality control that preserves interval order."""

from __future__ import annotations

import numpy as np

from config import (
    HAMPEL_HALF_WINDOW,
    HAMPEL_N_SIGMA,
    RR_MAX_MS,
    RR_MIN_MS,
)


def _hampel_outlier_mask(
    values: np.ndarray,
    half_window: int,
    n_sigma: float,
) -> np.ndarray:
    """Return a Hampel outlier mask using finite local neighbors."""
    mask = np.zeros(values.size, dtype=bool)

    for i in range(values.size):
        if not np.isfinite(values[i]):
            continue

        left = max(0, i - half_window)
        right = min(values.size, i + half_window + 1)
        local = values[left:right]
        local = local[np.isfinite(local)]

        if local.size < 3:
            continue

        median = np.median(local)
        mad = np.median(np.abs(local - median))
        if mad <= 0 or not np.isfinite(mad):
            continue

        threshold = n_sigma * 1.4826 * mad
        if abs(values[i] - median) > threshold:
            mask[i] = True

    return mask


def clean_rr_intervals(
    rr_ms: np.ndarray,
    rr_min_ms: float = RR_MIN_MS,
    rr_max_ms: float = RR_MAX_MS,
    hampel_half_window: int = HAMPEL_HALF_WINDOW,
    hampel_n_sigma: float = HAMPEL_N_SIGMA,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Clean RR intervals while preserving their temporal sequence.

    Physiologically implausible and Hampel-detected intervals are marked, then
    replaced by linear interpolation across interval index. Deleting intervals
    would incorrectly make originally non-adjacent intervals become adjacent,
    which can bias RMSSD, pNN, Poincare, entropy, and spectral features.
    """
    rr = np.asarray(rr_ms, dtype=float).reshape(-1)
    n_total = int(rr.size)

    if n_total == 0:
        stats = {
            "Num_RR_Raw": 0,
            "Num_RR_Clean": 0,
            "RR_Invalid": 0,
            "RR_HampelOutliers": 0,
            "RR_Corrected": 0,
            "RR_CorrectedPercent": 0.0,
        }
        return rr.copy(), stats

    invalid_mask = (
        ~np.isfinite(rr)
        | (rr < rr_min_ms)
        | (rr > rr_max_ms)
    )

    work = rr.copy()
    work[invalid_mask] = np.nan

    hampel_mask = _hampel_outlier_mask(
        work,
        half_window=hampel_half_window,
        n_sigma=hampel_n_sigma,
    )

    correction_mask = invalid_mask | hampel_mask
    work[correction_mask] = np.nan

    finite_idx = np.flatnonzero(np.isfinite(work))
    all_idx = np.arange(n_total)

    if finite_idx.size >= 2:
        cleaned = np.interp(all_idx, finite_idx, work[finite_idx])
    elif finite_idx.size == 1:
        cleaned = np.full(n_total, work[finite_idx[0]], dtype=float)
    else:
        cleaned = np.full(n_total, np.nan, dtype=float)

    n_corrected = int(np.sum(correction_mask))
    stats = {
        "Num_RR_Raw": n_total,
        "Num_RR_Clean": int(np.sum(np.isfinite(cleaned))),
        "RR_Invalid": int(np.sum(invalid_mask)),
        "RR_HampelOutliers": int(np.sum(hampel_mask & ~invalid_mask)),
        "RR_Corrected": n_corrected,
        "RR_CorrectedPercent": 100.0 * n_corrected / max(n_total, 1),
    }
    return cleaned, stats


def rr_quality(rr_ms: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper returning only cleaned RR values."""
    cleaned, _ = clean_rr_intervals(rr_ms)
    return cleaned
