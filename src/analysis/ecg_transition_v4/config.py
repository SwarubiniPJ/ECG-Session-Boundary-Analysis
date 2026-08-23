from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ALL23: list[str] = [
    "ECG_Mean", "ECG_STD", "ECG_VAR", "ECG_RMS", "ECG_Skew", "ECG_Kurtosis",
    "MeanHR", "StdHR", "MinHR", "MaxHR", "MeanRR",
    "SDNN", "RMSSD", "SDSD", "MedianNN", "MADNN", "pNN20", "pNN50",
    "SD1", "SD2", "SD1_SD2", "CSI", "CVI",
]

LOG1P_FEATURES: set[str] = {
    "ECG_STD", "ECG_VAR", "ECG_RMS", "StdHR", "SDNN", "RMSSD", "SDSD",
    "MADNN", "pNN20", "pNN50", "SD1", "SD2", "CSI",
}

FEATURE_FAMILIES: dict[str, list[str]] = {
    "all23": list(ALL23),
    "heart_rate_rr": ["MeanHR", "StdHR", "MinHR", "MaxHR", "MeanRR"],
    "hrv": [
        "SDNN", "RMSSD", "SDSD", "MedianNN", "MADNN", "pNN20", "pNN50",
        "SD1", "SD2", "SD1_SD2", "CSI", "CVI",
    ],
    "ecg_morphology": [
        "ECG_Mean", "ECG_STD", "ECG_VAR", "ECG_RMS", "ECG_Skew", "ECG_Kurtosis",
    ],
}

# Algebraically derived variables are removed before empirical correlation pruning.
REDUCED_DIRECT_CANDIDATES: list[str] = [
    "ECG_Mean", "ECG_STD", "ECG_Skew", "ECG_Kurtosis",
    "MeanHR", "MeanRR", "StdHR", "MinHR", "MaxHR",
    "SDNN", "RMSSD", "SDSD", "MedianNN", "MADNN", "pNN20", "pNN50",
]

REQUIRED_COLUMNS: set[str] = {
    "WindowID", "Subject", "Session", "ConditionCode", "Condition",
    "WindowLength_sec", "StepSize_sec", "StartTime_sec", "EndTime_sec",
    "CenterTime_sec", "SessionDuration_sec", "WindowIndex", "WindowValid",
    "RR_CorrectedPercent",
}

NEUTRAL_DIRECTION_LABELS: dict[str, str] = {
    "NA_to_A": "NA-to-A video-session boundary",
    "A_to_NA": "A-to-NA video-session boundary",
}

ALLOWED_WINDOWS: tuple[int, ...] = (30, 45, 60)

SEARCH_WINDOWS: dict[str, tuple[float, float]] = {
    "post_only": (0.0, 60.0),
    "anticipatory": (-30.0, 60.0),
}

SCORE_METHODS: tuple[str, ...] = (
    "LegacyBIC_fixed6",
    "CovIC",
    "BinSeg_L2",
    "SegmentedTrend",
    "CUSUM",
    "MOSUM",
)

PELT_MODELS: tuple[str, ...] = ("l2", "l1", "rbf")

TIMING_ENDPOINTS: tuple[str, ...] = (
    "departure_magnitude",
    "signed_trajectory",
)


@dataclass(frozen=True)
class RunConfig:
    input_csv: Path
    output_root: Path
    windows: tuple[int, ...] = ALLOWED_WINDOWS
    step_sec: int = 5
    pre_sec: float = 60.0
    post_sec: float = 60.0
    rr_thresholds: tuple[float, ...] = (5.0, 10.0, 20.0)
    min_segment_points: int = 5
    stable_edge_sec: float = 60.0
    reduced_abs_correlation: float = 0.90
    pca_variance_target: float = 0.80
    pseudo_controls_per_boundary: int = 50
    pseudo_block_separation_sec: float = 120.0
    pseudo_crossfit_folds: int = 4
    calibration_fpr: float = 0.05
    pelt_multipliers: tuple[float, ...] = (
        0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
        32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
    )
    bootstrap_replicates: int = 5000
    permutation_replicates: int = 10000
    null_simulations: int = 400
    power_simulations: int = 100
    simulation_representations: tuple[str, ...] = ("reduced", "independent_pca")

    # Participant-pooled timing analysis. This layer always reports a candidate
    # time for both directions, while keeping validation against matched stable
    # pseudo-boundaries separate from the descriptive timing estimate.
    run_population_timing: bool = True
    timing_windows: tuple[int, ...] = ALLOWED_WINDOWS
    timing_rr_thresholds: tuple[float, ...] = (20.0,)
    timing_representations: tuple[str, ...] = ("reduced", "independent_pca")
    timing_endpoints: tuple[str, ...] = ("departure_magnitude",)
    timing_search_windows: tuple[str, ...] = ("post_only", "anticipatory")
    timing_pseudo_draws: int = 1000
    timing_bootstrap_replicates: int = 2000
    timing_power_simulations: int = 50
    timing_simulation_effect_sizes: tuple[float, ...] = (0.5, 1.0, 1.5)
    timing_simulation_affected_fraction: float = 0.50
    timing_min_unique_times_per_side: int = 3
    timing_min_participants: int = 10
    timing_alpha: float = 0.05
    timing_primary_window: int = 30
    timing_primary_rr_threshold: float = 20.0
    timing_primary_representation: str = "reduced"
    timing_primary_endpoint: str = "departure_magnitude"
    timing_lopo: bool = True

    run_lopo: bool = True
    lopo_rr_thresholds: tuple[float, ...] = (20.0,)
    lopo_methods: tuple[str, ...] = (
        "CovIC", "SegmentedTrend", "CUSUM", "MOSUM", "PELT_L2"
    )
    seed: int = 42
    figure_dpi: int = 400
    require_ruptures: bool = False

    def validate(self) -> None:
        invalid = sorted(set(self.windows).difference(ALLOWED_WINDOWS))
        if invalid:
            raise ValueError(
                f"Only 30-, 45-, and 60-second windows are supported; received {invalid}."
            )
        if self.step_sec != 5:
            raise ValueError("This study uses a 5-second step; pass --step 5.")
        if self.pseudo_crossfit_folds < 2:
            raise ValueError("At least two pseudo cross-fitting folds are required.")
        if self.pseudo_controls_per_boundary < 1:
            raise ValueError("pseudo_controls_per_boundary must be positive.")
        if not 0.0 < self.calibration_fpr < 0.5:
            raise ValueError("calibration_fpr must be between 0 and 0.5.")
        invalid_timing_windows = sorted(set(self.timing_windows).difference(ALLOWED_WINDOWS))
        if invalid_timing_windows:
            raise ValueError(
                "Population timing supports only 30-, 45-, and 60-second windows; "
                f"received {invalid_timing_windows}."
            )
        missing_timing_windows = sorted(set(self.timing_windows).difference(self.windows))
        if missing_timing_windows:
            raise ValueError(
                "timing_windows must be a subset of the main --windows selection; "
                f"missing from main analysis: {missing_timing_windows}."
            )
        missing_timing_quality = sorted(
            set(float(value) for value in self.timing_rr_thresholds).difference(
                set(float(value) for value in self.rr_thresholds)
            )
        )
        if missing_timing_quality:
            raise ValueError(
                "timing_rr_thresholds must be a subset of the main RR thresholds; "
                f"missing from main analysis: {missing_timing_quality}."
            )
        invalid_search = sorted(set(self.timing_search_windows).difference(SEARCH_WINDOWS))
        if invalid_search:
            raise ValueError(f"Unknown population timing search windows: {invalid_search}")
        invalid_endpoints = sorted(set(self.timing_endpoints).difference(TIMING_ENDPOINTS))
        if invalid_endpoints:
            raise ValueError(f"Unknown population timing endpoints: {invalid_endpoints}")
        allowed_representations = {
            "all23", "reduced", "heart_rate_rr", "hrv",
            "ecg_morphology", "independent_pca",
        }
        invalid_representations = sorted(
            set(self.timing_representations).difference(allowed_representations)
        )
        if invalid_representations:
            raise ValueError(
                f"Unknown population timing representations: {invalid_representations}"
            )
        if self.timing_pseudo_draws < 100:
            raise ValueError("timing_pseudo_draws must be at least 100.")
        if self.timing_bootstrap_replicates < 100:
            raise ValueError("timing_bootstrap_replicates must be at least 100.")
        if self.timing_power_simulations < 0:
            raise ValueError("timing_power_simulations cannot be negative.")
        if any(value <= 0 for value in self.timing_simulation_effect_sizes):
            raise ValueError("timing_simulation_effect_sizes must be positive.")
        if not 0.0 < self.timing_simulation_affected_fraction <= 1.0:
            raise ValueError(
                "timing_simulation_affected_fraction must be in (0, 1]."
            )
        if self.timing_min_unique_times_per_side < 2:
            raise ValueError("timing_min_unique_times_per_side must be at least 2.")
        if self.timing_min_participants < 3:
            raise ValueError("timing_min_participants must be at least 3.")
        if not 0.0 < self.timing_alpha < 0.5:
            raise ValueError("timing_alpha must be between 0 and 0.5.")
        if self.timing_primary_window not in self.timing_windows:
            raise ValueError("timing_primary_window must be included in timing_windows.")
        if self.timing_primary_rr_threshold not in self.timing_rr_thresholds:
            raise ValueError(
                "timing_primary_rr_threshold must be included in timing_rr_thresholds."
            )
        if self.timing_primary_representation not in self.timing_representations:
            raise ValueError(
                "timing_primary_representation must be included in timing_representations."
            )
        if self.timing_primary_endpoint not in self.timing_endpoints:
            raise ValueError("timing_primary_endpoint must be included in timing_endpoints.")


@dataclass
class RepresentationSpec:
    name: str
    source_columns: list[str]
    output_columns: list[str]
    covariance: NDArray[np.float64]
    inverse_covariance: NDArray[np.float64]
    whitening: NDArray[np.float64]
    lag1_rho: float
    stable_rows: int
    scaler: Any | None = None
    pca: Any | None = None

    def transform(self, frame: Any) -> NDArray[np.float64]:
        x = frame[self.source_columns].to_numpy(dtype=float)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if self.pca is not None:
            x = self.pca.transform(x)
        return np.asarray(x, dtype=float)


@dataclass
class NormalizationParameters:
    subject: str
    excluded_sessions: tuple[int, ...]
    rows_a: int
    rows_na: int
    source: str
    centers: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    scale_methods: dict[str, str] = field(default_factory=dict)
