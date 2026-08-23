"""Feature extraction for one ECG window.

R-peaks can be supplied by the caller. The recommended pipeline detects peaks
once on the complete session and passes the peaks that fall inside each window.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

try:
    import neurokit2 as nk
except ImportError:  # A clear error is raised only when peak detection is used.
    nk = None

try:
    import antropy as ant
except ImportError:
    ant = None

from config import (
    CORRECT_RPEAK_ARTIFACTS,
    ECG_IS_FILTERED,
    MAX_RR_CORRECTED_PERCENT,
    MIN_RR_INTERVALS_FOR_DFA_ALPHA1,
    MIN_RR_INTERVALS_FOR_FREQUENCY,
    MIN_RR_INTERVALS_FOR_SAMPLE_ENTROPY,
    MIN_RR_INTERVALS_FOR_VALID_WINDOW,
    MIN_WINDOW_SEC_FOR_FREQUENCY,
    RPEAK_METHOD,
)
from signal_quality import clean_rr_intervals


ECG_FEATURE_NAMES = [
    "ECG_Mean",
    "ECG_STD",
    "ECG_VAR",
    "ECG_RMS",
    "ECG_Skew",
    "ECG_Kurtosis",
]

HR_FEATURE_NAMES = [
    "MeanHR",
    "StdHR",
    "MinHR",
    "MaxHR",
    "MeanRR",
]

TIME_FEATURE_NAMES = [
    "SDNN",
    "RMSSD",
    "SDSD",
    "MedianNN",
    "MADNN",
    "pNN20",
    "pNN50",
]

FREQUENCY_FEATURE_NAMES = [
    "VLF",
    "LF",
    "HF",
    "TotalPower",
    "LFHF",
    "LFnu",
    "HFnu",
]

NONLINEAR_FEATURE_NAMES = [
    "SD1",
    "SD2",
    "SD1_SD2",
    "SampEn",
    "DFA_alpha1",
    "CSI",
    "CVI",
]

ALL_FEATURE_NAMES = (
    ECG_FEATURE_NAMES
    + HR_FEATURE_NAMES
    + TIME_FEATURE_NAMES
    + FREQUENCY_FEATURE_NAMES
    + NONLINEAR_FEATURE_NAMES
)

QUALITY_FIELD_NAMES = [
    "WindowDuration_sec",
    "Num_RPeaks",
    "Num_RR_Raw",
    "Num_RR_Clean",
    "RR_Invalid",
    "RR_HampelOutliers",
    "RR_Corrected",
    "RR_CorrectedPercent",
    "WindowValid",
    "WindowFailureReason",
    "FrequencyFeaturesComputed",
    "FrequencyFeaturesStandard5Min",
    "EntropyFeatureComputed",
    "DFAFeatureComputed",
    "FeatureExtractionError",
]


def _nan_features(names: Iterable[str]) -> dict[str, float]:
    return {name: np.nan for name in names}


def _base_result(window_duration_sec: float) -> dict[str, object]:
    result: dict[str, object] = _nan_features(ALL_FEATURE_NAMES)
    result.update(
        {
            "WindowDuration_sec": float(window_duration_sec),
            "Num_RPeaks": 0,
            "Num_RR_Raw": 0,
            "Num_RR_Clean": 0,
            "RR_Invalid": 0,
            "RR_HampelOutliers": 0,
            "RR_Corrected": 0,
            "RR_CorrectedPercent": 0.0,
            "WindowValid": False,
            "WindowFailureReason": "",
            "FrequencyFeaturesComputed": False,
            "FrequencyFeaturesStandard5Min": window_duration_sec >= 300.0,
            "EntropyFeatureComputed": False,
            "DFAFeatureComputed": False,
            "FeatureExtractionError": "",
        }
    )
    return result


def ecg_statistics(ecg: np.ndarray) -> dict[str, float]:
    """Calculate simple amplitude-domain statistics."""
    values = np.asarray(ecg, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("ECG window has no finite samples")

    std = float(np.std(values, ddof=0))
    features = {
        "ECG_Mean": float(np.mean(values)),
        "ECG_STD": std,
        "ECG_VAR": float(np.var(values, ddof=0)),
        "ECG_RMS": float(np.sqrt(np.mean(np.square(values)))),
        "ECG_Skew": np.nan,
        "ECG_Kurtosis": np.nan,
    }

    # scipy returns undefined values for a constant signal. Leaving them as NaN
    # is more honest than silently replacing them by zero.
    if std > 0 and values.size >= 3:
        features["ECG_Skew"] = float(skew(values, bias=False))
    if std > 0 and values.size >= 4:
        features["ECG_Kurtosis"] = float(kurtosis(values, fisher=True, bias=False))

    return features


def detect_rpeaks(
    ecg: np.ndarray,
    fs: float,
    *,
    ecg_is_filtered: bool = ECG_IS_FILTERED,
) -> np.ndarray:
    """Detect R-peaks from a complete ECG session."""
    if nk is None:
        raise ImportError(
            "neurokit2 is required. Install dependencies with: "
            "python -m pip install neurokit2 antropy"
        )

    values = np.asarray(ecg, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("R-peak detection requires a non-empty finite ECG array")

    signal_for_detection = values
    if not ecg_is_filtered:
        signal_for_detection = nk.ecg_clean(
            values,
            sampling_rate=fs,
            method=RPEAK_METHOD,
        )

    _, info = nk.ecg_peaks(
        signal_for_detection,
        sampling_rate=fs,
        method=RPEAK_METHOD,
        correct_artifacts=CORRECT_RPEAK_ARTIFACTS,
    )

    rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    rpeaks = np.unique(rpeaks[(rpeaks >= 0) & (rpeaks < values.size)])
    return rpeaks


def heart_rate_features(rr_ms: np.ndarray) -> dict[str, float]:
    features = _nan_features(HR_FEATURE_NAMES)
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size == 0:
        return features

    hr = 60000.0 / rr
    features.update(
        {
            "MeanHR": float(np.mean(hr)),
            "StdHR": float(np.std(hr, ddof=0)),
            "MinHR": float(np.min(hr)),
            "MaxHR": float(np.max(hr)),
            "MeanRR": float(np.mean(rr)),
        }
    )
    return features


def time_domain_features(rr_ms: np.ndarray) -> dict[str, float]:
    features = _nan_features(TIME_FEATURE_NAMES)
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]

    if rr.size >= 2:
        median = float(np.median(rr))
        features["SDNN"] = float(np.std(rr, ddof=1))
        features["MedianNN"] = median
        features["MADNN"] = float(np.median(np.abs(rr - median)))

        diff_rr = np.diff(rr)
        features["RMSSD"] = float(np.sqrt(np.mean(np.square(diff_rr))))
        features["pNN20"] = float(100.0 * np.mean(np.abs(diff_rr) > 20.0))
        features["pNN50"] = float(100.0 * np.mean(np.abs(diff_rr) > 50.0))

        if diff_rr.size >= 2:
            features["SDSD"] = float(np.std(diff_rr, ddof=1))

    return features


def _band_power(frequency: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (frequency >= low) & (frequency < high)
    if np.sum(mask) < 2:
        return np.nan
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(psd[mask], frequency[mask]))


def frequency_domain_features(
    rr_ms: np.ndarray,
    window_duration_sec: float,
    interpolation_hz: float = 4.0,
) -> tuple[dict[str, float], bool]:
    """Calculate Welch HRV band powers in ms^2/Hz.

    Values are produced only when the configurable minimum duration and RR-count
    requirements are met. A 60-second threshold is still exploratory for LF/VLF;
    the output includes a separate 5-minute-standard flag.
    """
    features = _nan_features(FREQUENCY_FEATURE_NAMES)
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]

    if (
        window_duration_sec < MIN_WINDOW_SEC_FOR_FREQUENCY
        or rr.size < MIN_RR_INTERVALS_FOR_FREQUENCY
    ):
        return features, False

    # Associate each RR value with the start time of that interval.
    rr_time = np.concatenate(([0.0], np.cumsum(rr[:-1]) / 1000.0))
    if rr_time.size < 2 or rr_time[-1] <= 0:
        return features, False

    interpolation_time = np.arange(
        0.0,
        rr_time[-1] + 0.5 / interpolation_hz,
        1.0 / interpolation_hz,
    )
    if interpolation_time.size < 8:
        return features, False

    interpolated_rr = np.interp(interpolation_time, rr_time, rr)
    interpolated_rr = interpolated_rr - np.mean(interpolated_rr)

    nperseg = min(256, interpolated_rr.size)
    frequency, psd = welch(
        interpolated_rr,
        fs=interpolation_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    vlf = _band_power(frequency, psd, 0.0033, 0.04)
    lf = _band_power(frequency, psd, 0.04, 0.15)
    hf = _band_power(frequency, psd, 0.15, 0.40)
    total = _band_power(frequency, psd, 0.0033, 0.40)

    features.update({"VLF": vlf, "LF": lf, "HF": hf, "TotalPower": total})

    if np.isfinite(lf) and np.isfinite(hf) and hf > 0:
        features["LFHF"] = float(lf / hf)

    lf_hf_sum = lf + hf if np.isfinite(lf) and np.isfinite(hf) else np.nan
    if np.isfinite(lf_hf_sum) and lf_hf_sum > 0:
        features["LFnu"] = float(100.0 * lf / lf_hf_sum)
        features["HFnu"] = float(100.0 * hf / lf_hf_sum)

    return features, True


def poincare_features(rr_ms: np.ndarray) -> dict[str, float]:
    names = ["SD1", "SD2", "SD1_SD2", "CSI", "CVI"]
    features = _nan_features(names)
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]

    if rr.size < 3:
        return features

    diff_rr = np.diff(rr)
    sdsd = float(np.std(diff_rr, ddof=1))
    sdnn = float(np.std(rr, ddof=1))

    sd1 = sdsd / math.sqrt(2.0)
    sd2_squared = max(0.0, 2.0 * sdnn**2 - 0.5 * sdsd**2)
    sd2 = math.sqrt(sd2_squared)

    features["SD1"] = float(sd1)
    features["SD2"] = float(sd2)

    if sd2 > 0:
        features["SD1_SD2"] = float(sd1 / sd2)
    if sd1 > 0:
        features["CSI"] = float(sd2 / sd1)
    if sd1 > 0 and sd2 > 0:
        # Longitudinal axis = 4*SD2 and transverse axis = 4*SD1.
        features["CVI"] = float(np.log10(16.0 * sd1 * sd2))

    return features


def sample_entropy_feature(rr_ms: np.ndarray) -> tuple[float, bool]:
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size < MIN_RR_INTERVALS_FOR_SAMPLE_ENTROPY or ant is None:
        return np.nan, False

    try:
        value = float(ant.sample_entropy(rr))
        return (value, np.isfinite(value))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return np.nan, False


def dfa_alpha1_feature(rr_ms: np.ndarray) -> tuple[float, bool]:
    """Estimate short-scale DFA alpha1 over scales 4-16 RR intervals.

    NeuroKit2 is used instead of the external ``nolds`` package. This avoids
    the nolds 0.6.3 importlib.resources packaging bug while retaining an
    explicit, reproducible short-scale definition.
    """
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size < MIN_RR_INTERVALS_FOR_DFA_ALPHA1 or nk is None:
        return np.nan, False

    try:
        value, _ = nk.fractal_dfa(
            rr,
            scale=np.arange(4, 17),
            overlap=True,
            integrate=True,
            order=1,
            multifractal=False,
            show=False,
        )
        value = float(value)
        return value, bool(np.isfinite(value))
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        return np.nan, False


def extract_features(
    ecg: np.ndarray,
    fs: float,
    rpeaks: np.ndarray | None = None,
) -> dict[str, object]:
    """Extract ECG, HR, HRV, nonlinear, and quality features from one window.

    The function always returns a dictionary. When the window is not usable,
    feature fields remain NaN and quality fields explain why.
    """
    values = np.asarray(ecg, dtype=float).reshape(-1)
    duration_sec = values.size / float(fs)
    result = _base_result(duration_sec)

    try:
        result.update(ecg_statistics(values))

        if rpeaks is None:
            peaks = detect_rpeaks(values, fs)
        else:
            peaks = np.asarray(rpeaks, dtype=int).reshape(-1)
            peaks = np.unique(peaks[(peaks >= 0) & (peaks < values.size)])

        result["Num_RPeaks"] = int(peaks.size)

        rr_raw = np.diff(peaks).astype(float) / float(fs) * 1000.0
        rr_clean, rr_stats = clean_rr_intervals(rr_raw)
        result.update(rr_stats)

        failure_reasons: list[str] = []
        if not np.all(np.isfinite(values)):
            failure_reasons.append("nonfinite_ecg_samples")
        if float(result["ECG_STD"]) <= np.finfo(float).eps:
            failure_reasons.append("flat_ecg")
        if rr_raw.size < MIN_RR_INTERVALS_FOR_VALID_WINDOW:
            failure_reasons.append("too_few_rr_intervals")
        if not np.all(np.isfinite(rr_clean)):
            failure_reasons.append("rr_cleaning_failed")
        if float(result["RR_CorrectedPercent"]) > MAX_RR_CORRECTED_PERCENT:
            failure_reasons.append("too_many_rr_corrections")

        result["WindowValid"] = len(failure_reasons) == 0
        result["WindowFailureReason"] = ";".join(failure_reasons)

        if not result["WindowValid"]:
            return result

        result.update(heart_rate_features(rr_clean))
        result.update(time_domain_features(rr_clean))
        result.update(poincare_features(rr_clean))

        frequency_features, frequency_computed = frequency_domain_features(
            rr_clean,
            duration_sec,
        )
        result.update(frequency_features)
        result["FrequencyFeaturesComputed"] = frequency_computed

        sample_entropy, entropy_computed = sample_entropy_feature(rr_clean)
        result["SampEn"] = sample_entropy
        result["EntropyFeatureComputed"] = entropy_computed

        dfa_alpha1, dfa_computed = dfa_alpha1_feature(rr_clean)
        result["DFA_alpha1"] = dfa_alpha1
        result["DFAFeatureComputed"] = dfa_computed

        return result

    except Exception as exc:  # Preserve the window and expose the exact error.
        result["WindowValid"] = False
        result["WindowFailureReason"] = "feature_extraction_exception"
        result["FeatureExtractionError"] = f"{type(exc).__name__}: {exc}"
        return result
