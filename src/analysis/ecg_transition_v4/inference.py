from __future__ import annotations

import logging
import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import ALL23
from .normalization import quality_mask
from .utils import (
    benjamini_hochberg,
    normal_ci,
    participant_cluster_bootstrap,
    participant_sign_flip,
)

def cluster_bootstrap_mean_fast(
    data: pd.DataFrame,
    value_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = data[["Subject", value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan
    grouped = clean.groupby("Subject")[value_col].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    estimate = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sums), size=(n_bootstrap, len(sums)))
    boot_sums = sums[indices].sum(axis=1)
    boot_counts = counts[indices].sum(axis=1)
    boot = np.divide(
        boot_sums, boot_counts, out=np.full(n_bootstrap, np.nan), where=boot_counts > 0
    )
    low, high = np.nanquantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)


try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable
except Exception:  # pragma: no cover
    sm = None
    smf = None
    Exchangeable = None


def compute_feature_effects(
    boundaries: pd.DataFrame,
    window_sec: int,
    rr_threshold: float,
) -> pd.DataFrame:
    if boundaries.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for boundary_id, group in boundaries.groupby("BoundaryID", sort=False):
        first = group.iloc[0]
        pre = group[group["RelativeCenter_sec"].lt(0)]
        post = group[group["RelativeCenter_sec"].ge(0)]
        metadata = {
            "BoundaryID": boundary_id,
            "BoundaryKind": first["BoundaryKind"],
            "Subject": first["Subject"],
            "TransitionType": first.get("TransitionType", ""),
            "BoundaryOrder": int(first.get("BoundaryOrder", -1)),
            "PseudoCondition": first.get("PseudoCondition", ""),
            "CandidateID": first.get("CandidateID", ""),
            "CandidateBlockID": first.get("CandidateBlockID", ""),
            "PseudoFold": int(first.get("PseudoFold", -1)),
            "WindowLength_sec": window_sec,
            "RRThreshold": rr_threshold,
        }
        for feature in ALL23:
            zcol = f"{feature}_z"
            pre_raw = float(np.median(pre[feature].to_numpy(dtype=float)))
            post_raw = float(np.median(post[feature].to_numpy(dtype=float)))
            pre_z = float(np.median(pre[zcol].to_numpy(dtype=float)))
            post_z = float(np.median(post[zcol].to_numpy(dtype=float)))
            change = post_z - pre_z
            rows.append(
                {
                    **metadata,
                    "Feature": feature,
                    "PreMedianRaw": pre_raw,
                    "PostMedianRaw": post_raw,
                    "RawChange": post_raw - pre_raw,
                    "PreMedianZ": pre_z,
                    "PostMedianZ": post_z,
                    "ZChange": change,
                    "AbsoluteZChange": abs(change),
                    "Direction": (
                        "Increase" if change > 1e-12 else "Decrease" if change < -1e-12 else "No material change"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _deduplicate_for_rate(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return group
    if str(group["BoundaryKind"].iloc[0]) == "Real":
        return group.drop_duplicates("BoundaryID")
    keys = [
        "CandidateBlockID", "PseudoCondition", "TransitionType", "Method",
        "Representation", "SearchWindow", "WindowLength_sec", "RRThreshold",
    ]
    return group.sort_values("MatchScore" if "MatchScore" in group.columns else "BoundaryID").drop_duplicates(keys)


def summarize_detection_with_participant_ci(
    results: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group_columns = [
        "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow",
        "Method", "BoundaryKind", "TransitionType", "PseudoCondition",
    ]
    rows: list[dict[str, object]] = []
    for key_values, raw_group in results.groupby(group_columns, dropna=False, sort=False):
        key = dict(zip(group_columns, key_values))
        group = _deduplicate_for_rate(raw_group)
        if group.empty:
            continue

        group = group.assign(DetectedNumeric=group["Detected"].astype(float))
        rate, low, high = cluster_bootstrap_mean_fast(
            group, "DetectedNumeric", n_bootstrap, seed + len(rows) * 17
        )
        detected = group[group["Detected"] & group["Latency_sec"].notna()]

        magnitude, mag_low, mag_high = cluster_bootstrap_mean_fast(
            group, "MahalanobisMagnitude", n_bootstrap, seed + len(rows) * 19 + 3
        )
        rows.append(
            {
                **key,
                "Participants": group["Subject"].nunique(),
                "BoundariesOrPseudoBlocks": len(group),
                "Detections": int(group["Detected"].sum()),
                "DetectionRate": rate,
                "DetectionCI95_Lower": low,
                "DetectionCI95_Upper": high,
                "MedianLatencyDetected_sec": float(detected["Latency_sec"].median()) if len(detected) else np.nan,
                "Q1LatencyDetected_sec": float(detected["Latency_sec"].quantile(0.25)) if len(detected) else np.nan,
                "Q3LatencyDetected_sec": float(detected["Latency_sec"].quantile(0.75)) if len(detected) else np.nan,
                "MeanMahalanobisMagnitude": magnitude,
                "MagnitudeCI95_Lower": mag_low,
                "MagnitudeCI95_Upper": mag_high,
                "MedianScore": float(group["Score"].median()),
            }
        )
    return pd.DataFrame(rows)


def _matched_real_pseudo_summary(
    results: pd.DataFrame,
    pseudo_condition: str,
) -> pd.DataFrame:
    real = results[results["BoundaryKind"].eq("Real")].copy()
    pseudo = results[
        results["BoundaryKind"].eq("Pseudo")
        & results["PseudoCondition"].eq(pseudo_condition)
    ].copy()
    keys = [
        "MatchedRealBoundaryID", "Subject", "TransitionType", "BoundaryOrder",
        "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow", "Method",
    ]
    if "MatchedRealBoundaryID" not in real.columns:
        real["MatchedRealBoundaryID"] = real["BoundaryID"]
    real_agg = (
        real.groupby(keys, as_index=False)
        .agg(
            RealDetected=("Detected", "mean"),
            RealScore=("Score", "mean"),
            RealMagnitude=("MahalanobisMagnitude", "mean"),
            RealLatency=("Latency_sec", "mean"),
        )
    )
    pseudo_agg = (
        pseudo.groupby(keys, as_index=False)
        .agg(
            PseudoDetected=("Detected", "mean"),
            PseudoScore=("Score", "mean"),
            PseudoMagnitude=("MahalanobisMagnitude", "mean"),
            PseudoLatency=("Latency_sec", "mean"),
            PseudoControls=("CandidateBlockID", "nunique"),
        )
    )
    return real_agg.merge(pseudo_agg, on=keys, how="inner", validate="one_to_one")


def _paired_bootstrap_metrics(
    pairs: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if pairs.empty:
        return {name: np.nan for name in [
            "RealRate", "PseudoRate", "RiskDifference", "RiskDifferenceCI_Lower",
            "RiskDifferenceCI_Upper", "RiskRatio", "RiskRatioCI_Lower",
            "RiskRatioCI_Upper", "OddsRatio", "OddsRatioCI_Lower", "OddsRatioCI_Upper",
        ]}
    grouped = pairs.groupby("Subject").agg(
        RealSum=("RealDetected", "sum"),
        PseudoSum=("PseudoDetected", "sum"),
        Count=("MatchedRealBoundaryID", "size"),
    )
    real_sum = grouped["RealSum"].to_numpy(dtype=float)
    pseudo_sum = grouped["PseudoSum"].to_numpy(dtype=float)
    counts = grouped["Count"].to_numpy(dtype=float)

    def metrics(rs: np.ndarray, ps: np.ndarray, cs: np.ndarray) -> np.ndarray:
        total = float(cs.sum())
        real = float(rs.sum() / total)
        pseudo = float(ps.sum() / total)
        correction = 0.5 / total
        rr = (real + correction) / (pseudo + correction)
        odds_real = (real + correction) / (1.0 - real + correction)
        odds_pseudo = (pseudo + correction) / (1.0 - pseudo + correction)
        return np.asarray([real, pseudo, real - pseudo, rr, odds_real / odds_pseudo])

    estimate = metrics(real_sum, pseudo_sum, counts)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(counts), size=(n_bootstrap, len(counts)))
    boot_real_sum = real_sum[indices].sum(axis=1)
    boot_pseudo_sum = pseudo_sum[indices].sum(axis=1)
    boot_count = counts[indices].sum(axis=1)
    boot_real = boot_real_sum / boot_count
    boot_pseudo = boot_pseudo_sum / boot_count
    correction = 0.5 / boot_count
    boot_rd = boot_real - boot_pseudo
    boot_rr = (boot_real + correction) / (boot_pseudo + correction)
    boot_or = (
        (boot_real + correction) / (1.0 - boot_real + correction)
    ) / (
        (boot_pseudo + correction) / (1.0 - boot_pseudo + correction)
    )
    boot = np.column_stack([boot_real, boot_pseudo, boot_rd, boot_rr, boot_or])
    low = np.nanquantile(boot, 0.025, axis=0)
    high = np.nanquantile(boot, 0.975, axis=0)
    return {
        "RealRate": estimate[0],
        "PseudoRate": estimate[1],
        "RiskDifference": estimate[2],
        "RiskDifferenceCI_Lower": low[2],
        "RiskDifferenceCI_Upper": high[2],
        "RiskRatio": estimate[3],
        "RiskRatioCI_Lower": low[3],
        "RiskRatioCI_Upper": high[3],
        "OddsRatio": estimate[4],
        "OddsRatioCI_Lower": low[4],
        "OddsRatioCI_Upper": high[4],
    }


def compare_real_pseudo(
    results: pd.DataFrame,
    n_bootstrap: int,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_columns = [
        "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow",
        "Method", "TransitionType",
    ]
    comparison_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    for condition in ["A", "NA"]:
        logging.info("Real-pseudo comparison: preparing pseudo condition %s", condition)
        pairs = _matched_real_pseudo_summary(results, condition)
        grouped_pairs = list(pairs.groupby(group_columns, dropna=False, sort=False))
        logging.info(
            "Real-pseudo comparison: %s groups for pseudo condition %s",
            len(grouped_pairs), condition,
        )
        for group_number, (key_values, group) in enumerate(grouped_pairs, start=1):
            if group_number == 1 or group_number % 50 == 0:
                logging.info(
                    "Real-pseudo comparison group %s/%s (%s)",
                    group_number, len(grouped_pairs), condition,
                )
            key = dict(zip(group_columns, key_values))
            effects = _paired_bootstrap_metrics(
                group, n_bootstrap, seed + len(comparison_rows) * 23
            )
            participant = group.assign(
                DetectionDifference=group["RealDetected"] - group["PseudoDetected"],
                ScoreDifference=group["RealScore"] - group["PseudoScore"],
                MagnitudeDifference=group["RealMagnitude"] - group["PseudoMagnitude"],
            ).groupby("Subject")[[
                "DetectionDifference", "ScoreDifference", "MagnitudeDifference"
            ]].mean()
            d_obs, d_p, n = participant_sign_flip(
                participant["DetectionDifference"], permutations,
                seed + len(comparison_rows) * 29 + 1,
            )
            s_obs, s_p, _ = participant_sign_flip(
                participant["ScoreDifference"], permutations,
                seed + len(comparison_rows) * 31 + 2,
            )
            m_obs, m_p, _ = participant_sign_flip(
                participant["MagnitudeDifference"], permutations,
                seed + len(comparison_rows) * 37 + 3,
            )
            comparison_rows.append(
                {
                    **key,
                    "PseudoComparison": condition,
                    "Participants": n,
                    "MatchedRealBoundaries": group["MatchedRealBoundaryID"].nunique(),
                    **effects,
                    "MeanScoreDifference": float(group["RealScore"].sub(group["PseudoScore"]).mean()),
                    "MeanMagnitudeDifference": float(group["RealMagnitude"].sub(group["PseudoMagnitude"]).mean()),
                }
            )
            for outcome, observed, pvalue in [
                ("Detection rate", d_obs, d_p),
                ("Detector score", s_obs, s_p),
                ("Mahalanobis magnitude", m_obs, m_p),
            ]:
                permutation_rows.append(
                    {
                        **key,
                        "PseudoComparison": condition,
                        "Outcome": outcome,
                        "ObservedDifference": observed,
                        "PermutationPValue": pvalue,
                        "Participants": n,
                    }
                )
    return pd.DataFrame(comparison_rows), pd.DataFrame(permutation_rows)


def summarize_feature_effects(
    effects: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    group_columns = [
        "WindowLength_sec", "RRThreshold", "BoundaryKind", "TransitionType",
        "PseudoCondition", "Feature",
    ]
    rows: list[dict[str, object]] = []
    for key_values, raw_group in effects.groupby(group_columns, dropna=False, sort=False):
        key = dict(zip(group_columns, key_values))
        group = raw_group
        if key["BoundaryKind"] == "Pseudo":
            group = group.drop_duplicates(["CandidateBlockID", "Feature"])

        estimate, low, high = cluster_bootstrap_mean_fast(
            group, "ZChange", n_bootstrap, seed + len(rows) * 11
        )
        rows.append(
            {
                **key,
                "Participants": group["Subject"].nunique(),
                "BoundariesOrBlocks": len(group),
                "MeanZChange": estimate,
                "MeanZChangeCI95_Lower": low,
                "MeanZChangeCI95_Upper": high,
                "MedianZChange": float(group["ZChange"].median()),
                "MedianRawChange": float(group["RawChange"].median()),
                "MedianAbsoluteZChange": float(group["AbsoluteZChange"].median()),
                "IncreasePercent": 100.0 * float(group["ZChange"].gt(0).mean()),
                "DecreasePercent": 100.0 * float(group["ZChange"].lt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def quality_sensitivity_tables(
    data: pd.DataFrame,
    windows: Sequence[int],
    step_sec: int,
    thresholds: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for window in windows:
        base = data[
            data["WindowLength_sec"].eq(window)
            & data["StepSize_sec"].eq(step_sec)
        ].copy()
        for threshold in thresholds:
            base_valid = base["WindowValid"].fillna(False).astype(bool)
            passing = base_valid & base["RR_CorrectedPercent"].le(threshold)
            for condition, group in base.groupby("ConditionCode", dropna=False):
                valid = group["WindowValid"].fillna(False).astype(bool)
                group_pass = valid & group["RR_CorrectedPercent"].le(threshold)
                correction = group["RR_CorrectedPercent"].to_numpy(dtype=float)
                window_rows.append(
                    {
                        "WindowLength_sec": window,
                        "RRThreshold": threshold,
                        "ConditionCode": condition,
                        "TotalWindows": len(group),
                        "BaseValidWindows": int(valid.sum()),
                        "ThresholdPassingWindows": int(group_pass.sum()),
                        "ThresholdExcludedWindows": int(len(group) - group_pass.sum()),
                        "ThresholdPassPercent": 100.0 * float(group_pass.mean()),
                        "MeanRRCorrectedPercent": float(np.nanmean(correction)),
                        "MedianRRCorrectedPercent": float(np.nanmedian(correction)),
                        "Q1RRCorrectedPercent": float(np.nanquantile(correction, 0.25)),
                        "Q3RRCorrectedPercent": float(np.nanquantile(correction, 0.75)),
                        "WindowsAbove5Percent": int(np.sum(correction > 5)),
                        "WindowsAbove10Percent": int(np.sum(correction > 10)),
                        "WindowsAbove20Percent": int(np.sum(correction > 20)),
                        "TotalRRIntervals": float(group.get("Num_RR_Raw", pd.Series(dtype=float)).sum()),
                        "TotalRRCorrected": float(group.get("RR_Corrected", pd.Series(dtype=float)).sum()),
                    }
                )
            for subject, group in base.groupby("Subject", sort=True):
                valid = group["WindowValid"].fillna(False).astype(bool)
                group_pass = valid & group["RR_CorrectedPercent"].le(threshold)
                participant_rows.append(
                    {
                        "WindowLength_sec": window,
                        "RRThreshold": threshold,
                        "Subject": subject,
                        "TotalWindows": len(group),
                        "PassingWindows": int(group_pass.sum()),
                        "PassPercent": 100.0 * float(group_pass.mean()),
                        "MedianRRCorrectedPercent": float(group["RR_CorrectedPercent"].median()),
                    }
                )
            excluded = base.loc[~passing].copy()
            excluded["SensitivityFailureReason"] = np.where(
                ~base_valid.loc[~passing],
                excluded.get("WindowFailureReason", pd.Series(index=excluded.index, dtype=object)).fillna("base_window_invalid"),
                f"RR_CorrectedPercent>{threshold:g}",
            )
            for reason, count in excluded["SensitivityFailureReason"].fillna("unspecified").value_counts().items():
                failure_rows.append(
                    {
                        "WindowLength_sec": window,
                        "RRThreshold": threshold,
                        "FailureReason": reason,
                        "Windows": int(count),
                    }
                )
    return pd.DataFrame(window_rows), pd.DataFrame(participant_rows), pd.DataFrame(failure_rows)


def summarize_boundary_order(
    results: pd.DataFrame,
    primary_method: str,
    primary_representation: str,
    primary_search: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    real = results[
        results["BoundaryKind"].eq("Real")
        & results["Method"].eq(primary_method)
        & results["Representation"].eq(primary_representation)
        & results["SearchWindow"].eq(primary_search)
    ].copy()
    rows: list[dict[str, object]] = []
    for key_values, group in real.groupby(
        ["WindowLength_sec", "RRThreshold", "BoundaryOrder", "TransitionType"],
        dropna=False,
        sort=False,
    ):
        window, threshold, order, direction = key_values

        group = group.assign(DetectedNumeric=group["Detected"].astype(float))
        estimate, low, high = cluster_bootstrap_mean_fast(
            group, "DetectedNumeric", n_bootstrap, seed + len(rows) * 13
        )
        mag, mag_low, mag_high = cluster_bootstrap_mean_fast(
            group, "MahalanobisMagnitude", n_bootstrap, seed + len(rows) * 17
        )
        rows.append(
            {
                "WindowLength_sec": window,
                "RRThreshold": threshold,
                "BoundaryOrder": order,
                "TransitionType": direction,
                "Participants": group["Subject"].nunique(),
                "DetectionRate": estimate,
                "DetectionCI95_Lower": low,
                "DetectionCI95_Upper": high,
                "MeanMagnitude": mag,
                "MagnitudeCI95_Lower": mag_low,
                "MagnitudeCI95_Upper": mag_high,
                "MedianLatencyDetected_sec": float(group.loc[group["Detected"], "Latency_sec"].median()),
            }
        )
    return pd.DataFrame(rows)


def coefficient_rows(result: object, model: str, outcome: str, metadata: dict[str, object]) -> list[dict[str, object]]:
    params = pd.Series(result.params)
    bse = pd.Series(result.bse, index=params.index)
    pvalues = pd.Series(result.pvalues, index=params.index)
    rows: list[dict[str, object]] = []
    for term, estimate in params.items():
        se = float(bse.get(term, np.nan))
        low, high = normal_ci(float(estimate), se) if np.isfinite(se) else (np.nan, np.nan)
        rows.append(
            {
                **metadata,
                "Model": model,
                "Outcome": outcome,
                "Term": term,
                "Estimate": float(estimate),
                "StdError": se,
                "CI95_Lower": low,
                "CI95_Upper": high,
                "PValue": float(pvalues.get(term, np.nan)),
                "N": int(getattr(result, "nobs", np.nan)),
            }
        )
    return rows


def compare_directions_participant_level(
    results: pd.DataFrame,
    primary_method: str,
    primary_representation: str,
    primary_search: str,
    n_bootstrap: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Paired participant-level direction contrasts with bootstrap CIs/sign flips."""
    real = results[
        results["BoundaryKind"].eq("Real")
        & results["Method"].eq(primary_method)
        & results["Representation"].eq(primary_representation)
        & results["SearchWindow"].eq(primary_search)
    ].copy()
    if real.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    def paired_stats(pivot: pd.DataFrame) -> tuple[float, float, float, float, int]:
        paired = pivot.reindex(columns=["NA_to_A", "A_to_NA"]).dropna()
        if paired.empty:
            return np.nan, np.nan, np.nan, np.nan, 0
        difference = (paired["A_to_NA"] - paired["NA_to_A"]).to_numpy(dtype=float)
        observed = float(np.mean(difference))
        indices = rng.integers(0, len(difference), size=(n_bootstrap, len(difference)))
        bootstrap_values = np.mean(difference[indices], axis=1)
        low, high = np.quantile(bootstrap_values, [0.025, 0.975])
        signs = rng.choice(
            np.array([-1.0, 1.0]), size=(permutations, len(difference)), replace=True
        )
        null = np.mean(signs * difference[None, :], axis=1)
        pvalue = float(
            (1.0 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1.0)
        )
        return observed, float(low), float(high), pvalue, len(difference)

    for (window, rr_threshold), group in real.groupby(
        ["WindowLength_sec", "RRThreshold"], sort=False
    ):
        specifications = [
            ("DetectedNumeric", "Detection probability", False),
            ("Latency_sec", "Latency among detected boundaries (seconds)", True),
            ("MahalanobisMagnitude", "Mahalanobis effect magnitude", False),
            ("MeanAbsStandardizedChange", "Mean absolute standardized change", False),
            ("SignedFirstDimensionChange", "Signed first-dimension change", False),
        ]
        working = group.copy()
        working["DetectedNumeric"] = working["Detected"].astype(float)
        for column, outcome, conditional in specifications:
            source = working
            if conditional:
                source = working[working["Detected"] & working[column].notna()].copy()
            if column not in source.columns or source.empty:
                estimate = low = high = pvalue = np.nan
                n_paired = 0
                direction_means = pd.Series(dtype=float)
            else:
                pivot = (
                    source.groupby(["Subject", "TransitionType"])[column]
                    .mean()
                    .unstack()
                )
                estimate, low, high, pvalue, n_paired = paired_stats(pivot)
                direction_means = pivot.mean(axis=0)
            rows.append(
                {
                    "WindowLength_sec": window,
                    "RRThreshold": rr_threshold,
                    "Representation": primary_representation,
                    "Method": primary_method,
                    "SearchWindow": primary_search,
                    "Outcome": outcome,
                    "Contrast": "A_to_NA minus NA_to_A",
                    "NA_to_A_Estimate": direction_means.get("NA_to_A", np.nan),
                    "A_to_NA_Estimate": direction_means.get("A_to_NA", np.nan),
                    "Difference": estimate,
                    "CI95_Lower": low,
                    "CI95_Upper": high,
                    "ParticipantSignFlipPValue": pvalue,
                    "ParticipantsWithBothDirections": n_paired,
                    "ConditionalOnDetection": conditional,
                }
            )
    return pd.DataFrame(rows)


def run_direction_models(
    results: pd.DataFrame,
    primary_method: str,
    primary_representation: str,
    primary_search: str,
) -> pd.DataFrame:
    if sm is None or smf is None or Exchangeable is None:
        return pd.DataFrame()
    real = results[
        results["BoundaryKind"].eq("Real")
        & results["Method"].eq(primary_method)
        & results["Representation"].eq(primary_representation)
        & results["SearchWindow"].eq(primary_search)
    ].copy()
    rows: list[dict[str, object]] = []
    for key_values, group in real.groupby(["WindowLength_sec", "RRThreshold"], sort=False):
        metadata = {
            "WindowLength_sec": key_values[0],
            "RRThreshold": key_values[1],
            "Representation": primary_representation,
            "Method": primary_method,
            "SearchWindow": primary_search,
        }
        data = group.copy()
        data["DetectedNumeric"] = data["Detected"].astype(int)
        data["BoundaryOrderCentered"] = data["BoundaryOrder"] - data["BoundaryOrder"].mean()
        data["TransitionType"] = pd.Categorical(
            data["TransitionType"], categories=["NA_to_A", "A_to_NA"]
        )
        model_specs = [
            (
                "DetectedNumeric ~ C(TransitionType) + BoundaryOrderCentered",
                "Detected",
                sm.families.Binomial(),
            ),
            (
                "MahalanobisMagnitude ~ C(TransitionType) + BoundaryOrderCentered",
                "MahalanobisMagnitude",
                sm.families.Gaussian(),
            ),
        ]
        detected = data[data["Detected"] & data["Latency_sec"].notna()].copy()
        if len(detected) >= 20 and detected["TransitionType"].nunique() == 2:
            model_specs.append(
                (
                    "Latency_sec ~ C(TransitionType) + BoundaryOrderCentered",
                    "Latency_detected_only",
                    sm.families.Gaussian(),
                )
            )
        for formula, outcome, family in model_specs:
            source = detected if outcome == "Latency_detected_only" else data
            if outcome == "Detected":
                event_counts = (
                    source.groupby("TransitionType", observed=True)["DetectedNumeric"]
                    .agg(["sum", "count"])
                )
                separation = (
                    source["DetectedNumeric"].nunique() < 2
                    or event_counts.empty
                    or bool(((event_counts["sum"] == 0) | (event_counts["sum"] == event_counts["count"])).any())
                )
                if separation:
                    rows.append(
                        {
                            **metadata,
                            "Model": "GEE_exchangeable",
                            "Outcome": outcome,
                            "Term": "MODEL_NOT_ESTIMABLE_SEPARATION",
                            "Estimate": np.nan,
                            "StdError": np.nan,
                            "CI95_Lower": np.nan,
                            "CI95_Upper": np.nan,
                            "PValue": np.nan,
                            "N": len(source),
                            "Error": "Detection outcome had zero/all events in at least one direction; use the participant-level bootstrap/sign-flip contrast.",
                        }
                    )
                    continue
            try:
                model = smf.gee(
                    formula,
                    groups="Subject",
                    data=source,
                    family=family,
                    cov_struct=Exchangeable(),
                )
                fitted = model.fit(maxiter=200)
                rows.extend(coefficient_rows(fitted, "GEE_exchangeable", outcome, metadata))
            except Exception as exc:
                rows.append(
                    {
                        **metadata,
                        "Model": "GEE_exchangeable",
                        "Outcome": outcome,
                        "Term": "MODEL_FAILED",
                        "Estimate": np.nan,
                        "StdError": np.nan,
                        "CI95_Lower": np.nan,
                        "CI95_Upper": np.nan,
                        "PValue": np.nan,
                        "N": len(source),
                        "Error": str(exc),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["QValue_BH"] = out.groupby(["Outcome", "Term"], group_keys=False)["PValue"].transform(benjamini_hochberg)
    return out


def run_feature_direction_models(effects: pd.DataFrame) -> pd.DataFrame:
    if sm is None or smf is None or Exchangeable is None or effects.empty:
        return pd.DataFrame()
    real = effects[effects["BoundaryKind"].eq("Real")].copy()
    rows: list[dict[str, object]] = []
    for key_values, group in real.groupby(
        ["WindowLength_sec", "RRThreshold", "Feature"], sort=False
    ):
        window, threshold, feature = key_values
        data = group.copy()
        data["BoundaryOrderCentered"] = data["BoundaryOrder"] - data["BoundaryOrder"].mean()
        data["TransitionType"] = pd.Categorical(
            data["TransitionType"], categories=["NA_to_A", "A_to_NA"]
        )
        try:
            model = smf.gee(
                "ZChange ~ C(TransitionType) + BoundaryOrderCentered",
                groups="Subject",
                data=data,
                family=sm.families.Gaussian(),
                cov_struct=Exchangeable(),
            )
            fitted = model.fit(maxiter=200)
            rows.extend(
                coefficient_rows(
                    fitted,
                    "GEE_exchangeable",
                    "Feature_ZChange",
                    {
                        "WindowLength_sec": window,
                        "RRThreshold": threshold,
                        "Feature": feature,
                    },
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "WindowLength_sec": window,
                    "RRThreshold": threshold,
                    "Feature": feature,
                    "Model": "GEE_exchangeable",
                    "Outcome": "Feature_ZChange",
                    "Term": "MODEL_FAILED",
                    "Estimate": np.nan,
                    "StdError": np.nan,
                    "CI95_Lower": np.nan,
                    "CI95_Upper": np.nan,
                    "PValue": np.nan,
                    "N": len(data),
                    "Error": str(exc),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["QValue_BH"] = out.groupby(
            ["WindowLength_sec", "RRThreshold", "Term"], group_keys=False
        )["PValue"].transform(benjamini_hochberg)
    return out
