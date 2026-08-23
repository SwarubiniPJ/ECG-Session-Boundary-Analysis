from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import RepresentationSpec, RunConfig, SEARCH_WINDOWS
from .detectors import (
    binseg_l2_score,
    covariance_ic_score,
    cusum_score,
    legacy_bic_score,
    mosum_score,
    pelt_result,
    segmented_trend_score,
)
from .utils import ensure_dir, quantile_higher


def stable_covariance_cholesky(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def simulate_ar1_sequence(
    n: int,
    covariance: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    d = covariance.shape[0]
    root = stable_covariance_cholesky(covariance)
    out = np.empty((n, d), dtype=float)
    out[0] = root @ rng.normal(size=d)
    innovation_scale = math.sqrt(max(1.0 - rho * rho, 1e-6))
    for index in range(1, n):
        innovation = innovation_scale * (root @ rng.normal(size=d))
        out[index] = rho * out[index - 1] + innovation
    return out


def inject_known_change(
    matrix: np.ndarray,
    split: int,
    covariance: np.ndarray,
    effect_size: float,
    affected_fraction: float,
    change_shape: str,
    rng: np.random.Generator,
) -> np.ndarray:
    out = matrix.copy()
    d = out.shape[1]
    affected = max(1, int(math.ceil(d * affected_fraction)))
    indices = rng.choice(np.arange(d), size=affected, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=affected)
    standard_deviations = np.sqrt(np.clip(np.diag(covariance), 1e-8, None))
    shift = np.zeros(d, dtype=float)
    shift[indices] = effect_size * standard_deviations[indices] * signs
    if change_shape == "abrupt":
        out[split:] += shift
    elif change_shape == "ramp":
        ramp = np.linspace(0.0, 1.0, len(out) - split, endpoint=True)[:, None]
        out[split:] += ramp * shift[None, :]
    else:
        raise ValueError(f"Unsupported change shape: {change_shape}")
    return out


def _score_methods(
    times: np.ndarray,
    matrix: np.ndarray,
    spec: RepresentationSpec,
    min_points: int,
    search_start: float,
    search_end: float,
) -> dict[str, dict[str, object]]:
    return {
        "LegacyBIC_fixed6": legacy_bic_score(
            times, matrix, min_points, search_start, search_end
        ),
        "CovIC": covariance_ic_score(
            times, matrix, spec, min_points, search_start, search_end
        ),
        "BinSeg_L2": binseg_l2_score(
            times, matrix, min_points, search_start, search_end
        ),
        "SegmentedTrend": segmented_trend_score(
            times, matrix, spec, min_points, search_start, search_end
        ),
        "CUSUM": cusum_score(
            times, matrix, spec, min_points, search_start, search_end
        ),
        "MOSUM": mosum_score(
            times, matrix, spec, min_points, search_start, search_end
        ),
    }


def run_simulation_calibration(
    window_sec: int,
    representation: str,
    spec: RepresentationSpec,
    time_templates: list[np.ndarray],
    config: RunConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(output_dir)
    rng = np.random.default_rng(config.seed + window_sec * 101 + len(representation))
    null_rows: list[dict[str, object]] = []
    pelt_null_rows: list[dict[str, object]] = []
    for simulation in range(config.null_simulations):
        times = np.asarray(
            time_templates[int(rng.integers(0, len(time_templates)))], dtype=float
        )
        matrix = simulate_ar1_sequence(
            len(times), spec.covariance, spec.lag1_rho, rng
        )
        for search_name, (search_start, search_end) in SEARCH_WINDOWS.items():
            results = _score_methods(
                times, matrix, spec, config.min_segment_points, search_start, search_end
            )
            for method, result in results.items():
                null_rows.append(
                    {
                        "WindowLength_sec": window_sec,
                        "Representation": representation,
                        "SearchWindow": search_name,
                        "Simulation": simulation,
                        "Method": method,
                        "Score": result.get("Score", np.nan),
                    }
                )
            for model in ["l2", "l1", "rbf"]:
                for multiplier in config.pelt_multipliers:
                    result = pelt_result(
                        times,
                        matrix,
                        spec,
                        model,
                        multiplier,
                        config.min_segment_points,
                        search_start,
                        search_end,
                        require_ruptures=False,
                    )
                    pelt_null_rows.append(
                        {
                            "WindowLength_sec": window_sec,
                            "Representation": representation,
                            "SearchWindow": search_name,
                            "Simulation": simulation,
                            "CostModel": model,
                            "PenaltyMultiplier": multiplier,
                            "Detected": result["DetectedAtPenalty"],
                            "Penalty": result["Penalty"],
                            "Score": result["Score"],
                        }
                    )

    null_scores = pd.DataFrame(null_rows)
    thresholds = (
        null_scores.groupby(
            ["WindowLength_sec", "Representation", "SearchWindow", "Method"],
            as_index=False,
        )["Score"]
        .agg(
            SimulationThreshold=lambda x: quantile_higher(x, 0.95),
            NullMedian="median",
            NullSimulations="size",
        )
    )
    thresholds["TargetFalsePositiveRate"] = 0.05
    thresholds["ObservedFalsePositiveRate"] = thresholds.apply(
        lambda row: float(
            np.mean(
                null_scores[
                    null_scores["WindowLength_sec"].eq(row.WindowLength_sec)
                    & null_scores["Representation"].eq(row.Representation)
                    & null_scores["SearchWindow"].eq(row.SearchWindow)
                    & null_scores["Method"].eq(row.Method)
                ]["Score"].to_numpy(dtype=float)
                >= row.SimulationThreshold
            )
        ),
        axis=1,
    )
    # Explicitly show the liberal false-positive rate of the legacy fixed rule.
    legacy = null_scores[null_scores["Method"].eq("LegacyBIC_fixed6")].copy()
    legacy_fixed = (
        legacy.groupby(
            ["WindowLength_sec", "Representation", "SearchWindow"], as_index=False
        )
        .agg(
            NullSimulations=("Score", "size"),
            Fixed6FalsePositiveRate=("Score", lambda x: float(np.mean(np.asarray(x) >= 6.0))),
        )
    )

    pelt_null = pd.DataFrame(pelt_null_rows)
    pelt_summary = (
        pelt_null.groupby(
            [
                "WindowLength_sec", "Representation", "SearchWindow",
                "CostModel", "PenaltyMultiplier",
            ],
            as_index=False,
        )
        .agg(
            NullSimulations=("Detected", "size"),
            FalsePositiveRate=("Detected", "mean"),
            MedianPenalty=("Penalty", "median"),
            MedianScore=("Score", "median"),
        )
    )

    threshold_lookup = thresholds.set_index(["SearchWindow", "Method"])[
        "SimulationThreshold"
    ].to_dict()
    selected_pelt: dict[tuple[str, str], float] = {}
    pelt_summary["Selected"] = False
    for (search_name, model), group in pelt_summary.groupby(
        ["SearchWindow", "CostModel"], sort=False
    ):
        acceptable = group[group["FalsePositiveRate"].le(0.05)].copy()
        if acceptable.empty:
            chosen = group.sort_values("PenaltyMultiplier").iloc[-1]
        else:
            acceptable["Distance"] = (acceptable["FalsePositiveRate"] - 0.05).abs()
            chosen = acceptable.sort_values(["Distance", "PenaltyMultiplier"]).iloc[0]
        selected_pelt[(search_name, model)] = float(chosen.PenaltyMultiplier)
        mask = (
            pelt_summary["SearchWindow"].eq(search_name)
            & pelt_summary["CostModel"].eq(model)
            & pelt_summary["PenaltyMultiplier"].eq(chosen.PenaltyMultiplier)
        )
        pelt_summary.loc[mask, "Selected"] = True
    pelt_summary["Selected"] = pelt_summary["Selected"].astype(bool)

    power_rows: list[dict[str, object]] = []
    effect_sizes = [0.25, 0.50, 0.75, 1.00, 1.50]
    affected_fractions = [0.25, 0.50, 1.00]
    positions = ["early", "middle", "late"]
    shapes = ["abrupt", "ramp"]
    for effect_size in effect_sizes:
        for fraction in affected_fractions:
            for position in positions:
                for shape in shapes:
                    for simulation in range(config.power_simulations):
                        times = np.asarray(
                            time_templates[int(rng.integers(0, len(time_templates)))],
                            dtype=float,
                        )
                        nonnegative = np.flatnonzero(times >= 0)
                        if nonnegative.size == 0:
                            continue
                        earliest = int(nonnegative[0])
                        latest = len(times) - config.min_segment_points
                        true_split = (
                            earliest
                            if position == "early"
                            else latest
                            if position == "late"
                            else int(round((earliest + latest) / 2))
                        )
                        true_split = int(
                            np.clip(
                                true_split,
                                config.min_segment_points,
                                len(times) - config.min_segment_points,
                            )
                        )
                        matrix = simulate_ar1_sequence(
                            len(times), spec.covariance, spec.lag1_rho, rng
                        )
                        matrix = inject_known_change(
                            matrix,
                            true_split,
                            spec.covariance,
                            effect_size,
                            fraction,
                            shape,
                            rng,
                        )
                        for search_name, (search_start, search_end) in SEARCH_WINDOWS.items():
                            results = _score_methods(
                                times,
                                matrix,
                                spec,
                                config.min_segment_points,
                                search_start,
                                search_end,
                            )
                            for method, result in results.items():
                                if method == "LegacyBIC_fixed6":
                                    threshold = 6.0
                                else:
                                    threshold = threshold_lookup.get((search_name, method), np.nan)
                                detected = bool(
                                    np.isfinite(result.get("Score", np.nan))
                                    and float(result["Score"]) >= threshold
                                )
                                estimate = (
                                    float(result.get("CandidateLatency_sec", np.nan))
                                    if detected
                                    else np.nan
                                )
                                power_rows.append(
                                    {
                                        "WindowLength_sec": window_sec,
                                        "Representation": representation,
                                        "SearchWindow": search_name,
                                        "EffectSizeSD": effect_size,
                                        "AffectedFraction": fraction,
                                        "ChangePosition": position,
                                        "ChangeShape": shape,
                                        "Simulation": simulation,
                                        "Method": method,
                                        "Detected": detected,
                                        "TrueChangeTime_sec": float(times[true_split]),
                                        "EstimatedChangeTime_sec": estimate,
                                        "AbsoluteTimingError_sec": (
                                            abs(estimate - float(times[true_split]))
                                            if detected
                                            else np.nan
                                        ),
                                    }
                                )
                            for model in ["l2", "l1", "rbf"]:
                                multiplier = selected_pelt[(search_name, model)]
                                result = pelt_result(
                                    times,
                                    matrix,
                                    spec,
                                    model,
                                    multiplier,
                                    config.min_segment_points,
                                    search_start,
                                    search_end,
                                    require_ruptures=False,
                                )
                                detected = bool(result["DetectedAtPenalty"])
                                estimate = (
                                    float(result["CandidateLatency_sec"])
                                    if detected
                                    else np.nan
                                )
                                power_rows.append(
                                    {
                                        "WindowLength_sec": window_sec,
                                        "Representation": representation,
                                        "SearchWindow": search_name,
                                        "EffectSizeSD": effect_size,
                                        "AffectedFraction": fraction,
                                        "ChangePosition": position,
                                        "ChangeShape": shape,
                                        "Simulation": simulation,
                                        "Method": f"PELT_{model.upper()}",
                                        "Detected": detected,
                                        "TrueChangeTime_sec": float(times[true_split]),
                                        "EstimatedChangeTime_sec": estimate,
                                        "AbsoluteTimingError_sec": (
                                            abs(estimate - float(times[true_split]))
                                            if detected
                                            else np.nan
                                        ),
                                    }
                                )

    power_raw = pd.DataFrame(power_rows)
    power_summary = (
        power_raw.groupby(
            [
                "WindowLength_sec", "Representation", "SearchWindow", "EffectSizeSD",
                "AffectedFraction", "ChangePosition", "ChangeShape", "Method",
            ],
            as_index=False,
        )
        .agg(
            Simulations=("Detected", "size"),
            DetectionPower=("Detected", "mean"),
            MedianAbsoluteTimingError_sec=("AbsoluteTimingError_sec", "median"),
            MeanAbsoluteTimingError_sec=("AbsoluteTimingError_sec", "mean"),
        )
    )

    null_scores.to_csv(
        output_dir / f"null_scores_{representation}_w{window_sec}.csv", index=False
    )
    thresholds.to_csv(
        output_dir / f"simulation_thresholds_{representation}_w{window_sec}.csv",
        index=False,
    )
    legacy_fixed.to_csv(
        output_dir / f"legacy_fixed6_null_fpr_{representation}_w{window_sec}.csv",
        index=False,
    )
    pelt_summary.to_csv(
        output_dir / f"pelt_simulation_penalty_grid_{representation}_w{window_sec}.csv",
        index=False,
    )
    power_summary.to_csv(
        output_dir / f"simulation_power_timing_{representation}_w{window_sec}.csv",
        index=False,
    )
    return thresholds, legacy_fixed, pelt_summary, power_summary
