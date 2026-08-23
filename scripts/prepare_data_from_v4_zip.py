#!/usr/bin/env python3
"""Prepare manuscript-ready figure and table datasets from the V4 result archive.

The script accepts either the V4 ZIP archive or an already-extracted V4 results
folder. It creates a compact, self-contained data directory used by
``generate_manuscript_outputs.py``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_METHOD = "CovIC_crossfit"
PRIMARY_REPRESENTATION = "reduced"
PRIMARY_RR_THRESHOLD = 20.0
PRIMARY_SEARCH = "post_only"
TIMING_ENDPOINT = "departure_magnitude"
TIMING_REPRESENTATIONS = ("reduced", "independent_pca")

METHOD_LABELS = {
    "LegacyBIC_fixed6": "Legacy BIC, fixed 6",
    "LegacyBIC_crossfit": "Legacy BIC, calibrated",
    "CovIC_crossfit": "Covariance-aware IC",
    "BinSeg_L2_crossfit": "Binary segmentation L2",
    "SegmentedTrend_crossfit": "Segmented trend",
    "CUSUM_crossfit": "CUSUM",
    "MOSUM_crossfit": "MOSUM",
    "PELT_L1_crossfit": "PELT L1",
    "PELT_L2_crossfit": "PELT L2",
    "PELT_RBF_crossfit": "PELT RBF",
}

METHOD_ORDER = list(METHOD_LABELS)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return result
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * ranked)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    restored = np.empty_like(adjusted_sorted)
    restored[order] = adjusted_sorted
    result[np.flatnonzero(finite)] = restored
    return result


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return result
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[np.flatnonzero(finite)] = restored
    return result


def resolve_root(source: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    source = source.expanduser().resolve()
    if source.is_dir():
        direct = source / "04_inference"
        if direct.exists():
            return source, None
        children = [p for p in source.iterdir() if p.is_dir() and (p / "04_inference").exists()]
        if len(children) == 1:
            return children[0], None
        raise FileNotFoundError(f"Could not identify V4 results root under {source}")

    if source.suffix.lower() != ".zip":
        raise ValueError("--source must be a V4 results directory or ZIP archive")
    temp = tempfile.TemporaryDirectory(prefix="v4_results_")
    with zipfile.ZipFile(source) as archive:
        archive.extractall(temp.name)
    base = Path(temp.name)
    candidates = [p for p in base.rglob("04_inference") if p.is_dir() and "__MACOSX" not in p.parts]
    if len(candidates) != 1:
        temp.cleanup()
        raise FileNotFoundError("Could not uniquely identify 04_inference in the ZIP")
    return candidates[0].parent, temp


def write_csv(df: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip:
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="V4 results ZIP or extracted folder")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    root, temp = resolve_root(args.source)
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "main").mkdir(parents=True)
    (out / "supplement").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)

    manifest: list[dict[str, str]] = []

    def record(output_name: str, source_name: str, filters: str, purpose: str) -> None:
        manifest.append(
            {
                "OutputFile": output_name,
                "V4SourceFile": source_name,
                "FiltersOrDerivation": filters,
                "Purpose": purpose,
            }
        )

    # ------------------------------------------------------------------
    # Study inventory and quality retention
    # ------------------------------------------------------------------
    inventory = read_csv(root / "00_audit/real_boundary_inventory.csv")
    common = read_csv(root / "05_sensitivity/common_real_boundaries_all_windows_quality.csv")
    study_inventory = pd.DataFrame(
        [
            ["Participants", inventory["Subject"].nunique()],
            ["Real session boundaries", len(inventory)],
            ["A-to-NA boundaries", int((inventory["TransitionType"] == "A_to_NA").sum())],
            ["NA-to-A boundaries", int((inventory["TransitionType"] == "NA_to_A").sum())],
            ["Common-sample boundaries", common["BoundaryID"].nunique()],
            ["Common-sample participants", common["Subject"].nunique()],
        ],
        columns=["Metric", "Value"],
    )
    write_csv(study_inventory, out / "tables/table_1_study_inventory.csv")
    record(
        "tables/table_1_study_inventory.csv",
        "00_audit/real_boundary_inventory.csv; 05_sensitivity/common_real_boundaries_all_windows_quality.csv",
        "Counts of participants, boundary directions and common-sample boundaries",
        "Main Table 1 inventory",
    )

    quality = to_numeric(
        read_csv(root / "05_sensitivity/rr_quality_window_summary.csv"),
        [
            "WindowLength_sec",
            "RRThreshold",
            "TotalWindows",
            "BaseValidWindows",
            "ThresholdPassingWindows",
            "ThresholdExcludedWindows",
            "TotalRRIntervals",
            "TotalRRCorrected",
        ],
    )
    quality_agg = (
        quality.groupby(["WindowLength_sec", "RRThreshold"], as_index=False)
        .agg(
            TotalWindows=("TotalWindows", "sum"),
            BaseValidWindows=("BaseValidWindows", "sum"),
            PassingWindows=("ThresholdPassingWindows", "sum"),
            ExcludedWindows=("ThresholdExcludedWindows", "sum"),
            TotalRRIntervals=("TotalRRIntervals", "sum"),
            CorrectedRRIntervals=("TotalRRCorrected", "sum"),
        )
    )
    quality_agg["WindowRetentionPercent"] = 100 * quality_agg["PassingWindows"] / quality_agg["TotalWindows"]
    quality_agg["RRCorrectionPercent"] = 100 * quality_agg["CorrectedRRIntervals"] / quality_agg["TotalRRIntervals"]

    eligibility = read_csv(root / "05_sensitivity/real_boundary_eligibility.csv")
    eligibility["Eligible"] = eligibility["Eligible"].astype(str).str.lower().eq("true")
    eligibility = to_numeric(eligibility, ["WindowLength_sec", "RRThreshold"])
    eligible_counts = (
        eligibility.groupby(["WindowLength_sec", "RRThreshold"], as_index=False)
        .agg(EligibleBoundaries=("Eligible", "sum"), TotalBoundaries=("BoundaryID", "nunique"))
    )
    quality_main = quality_agg.merge(eligible_counts, on=["WindowLength_sec", "RRThreshold"], how="left")
    quality_main["BoundaryRetentionPercent"] = 100 * quality_main["EligibleBoundaries"] / quality_main["TotalBoundaries"]
    write_csv(quality_main, out / "main/figure_5_quality_sensitivity.csv")
    write_csv(quality_main, out / "tables/table_1_quality_retention.csv")
    write_csv(quality, out / "supplement/quality_by_condition.csv")
    record(
        "main/figure_5_quality_sensitivity.csv",
        "05_sensitivity/rr_quality_window_summary.csv; 05_sensitivity/real_boundary_eligibility.csv",
        "Aggregated across A and NA sessions by window and RR threshold",
        "Quality-sensitivity line plots",
    )

    # ------------------------------------------------------------------
    # Primary magnitude asymmetry
    # ------------------------------------------------------------------
    eval_results = read_csv(root / "03_boundary_results/evaluation_results_crossfitted_v4.csv.gz")
    eval_results = to_numeric(
        eval_results,
        ["WindowLength_sec", "RRThreshold", "MahalanobisMagnitude", "MeanAbsStandardizedChange"],
    )
    primary_eval = eval_results[
        (eval_results["BoundaryKind"] == "Real")
        & (eval_results["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (eval_results["Representation"] == PRIMARY_REPRESENTATION)
        & (eval_results["SearchWindow"] == PRIMARY_SEARCH)
        & (eval_results["Method"] == PRIMARY_METHOD)
    ].copy()
    participant_points = (
        primary_eval.groupby(["WindowLength_sec", "Subject", "TransitionType"], as_index=False)
        .agg(
            ParticipantMeanMagnitude=("MahalanobisMagnitude", "mean"),
            ParticipantMeanAbsoluteChange=("MeanAbsStandardizedChange", "mean"),
            Boundaries=("BoundaryID", "nunique"),
        )
    )
    write_csv(participant_points, out / "main/figure_1_magnitude_participant_points.csv")
    record(
        "main/figure_1_magnitude_participant_points.csv",
        "03_boundary_results/evaluation_results_crossfitted_v4.csv.gz",
        "Real, RR<=20%, reduced, post-only, CovIC; participant means by direction",
        "Paired participant plot",
    )

    direction = read_csv(root / "04_inference/direction_comparison_participant_bootstrap.csv")
    direction = to_numeric(
        direction,
        [
            "WindowLength_sec",
            "RRThreshold",
            "NA_to_A_Estimate",
            "A_to_NA_Estimate",
            "Difference",
            "CI95_Lower",
            "CI95_Upper",
            "ParticipantSignFlipPValue",
        ],
    )
    mag_summary = direction[
        (direction["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (direction["Representation"] == PRIMARY_REPRESENTATION)
        & (direction["Method"] == PRIMARY_METHOD)
        & (direction["SearchWindow"] == PRIMARY_SEARCH)
        & direction["Outcome"].isin(["Mahalanobis effect magnitude", "Mean absolute standardized change"])
    ].copy()
    mag_summary["PValue_Holm3"] = np.nan
    for outcome, idx in mag_summary.groupby("Outcome").groups.items():
        mag_summary.loc[idx, "PValue_Holm3"] = holm_adjust(mag_summary.loc[idx, "ParticipantSignFlipPValue"])
    mag_summary["SupportedAfterHolm3"] = mag_summary["PValue_Holm3"] < 0.05
    write_csv(mag_summary, out / "main/figure_1_magnitude_summary.csv")
    write_csv(mag_summary, out / "tables/table_2_primary_magnitude.csv")
    record(
        "main/figure_1_magnitude_summary.csv",
        "04_inference/direction_comparison_participant_bootstrap.csv",
        "RR<=20%, reduced, CovIC, post-only; Holm correction separately over 3 windows for each magnitude outcome",
        "Primary forest plot and Table 2",
    )

    # ------------------------------------------------------------------
    # Real versus pseudo detection
    # ------------------------------------------------------------------
    detection = read_csv(root / "04_inference/detection_rates_participant_bootstrap.csv")
    detection = to_numeric(
        detection,
        [
            "WindowLength_sec",
            "RRThreshold",
            "DetectionRate",
            "DetectionCI95_Lower",
            "DetectionCI95_Upper",
            "Participants",
            "BoundariesOrPseudoBlocks",
            "Detections",
        ],
    )
    primary_detection = detection[
        (detection["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (detection["Representation"] == PRIMARY_REPRESENTATION)
        & (detection["SearchWindow"] == PRIMARY_SEARCH)
        & (detection["Method"] == PRIMARY_METHOD)
    ].copy()
    primary_detection["Group"] = np.where(
        primary_detection["BoundaryKind"].eq("Real"),
        "Real",
        "Pseudo " + primary_detection["PseudoCondition"].astype(str),
    )
    primary_detection["Group"] = primary_detection["Group"].replace({"Pseudo NA": "Pseudo NA", "Pseudo A": "Pseudo A"})
    write_csv(primary_detection, out / "main/figure_2_detection_rates.csv")
    write_csv(primary_detection, out / "tables/table_3_real_pseudo_detection.csv")
    record(
        "main/figure_2_detection_rates.csv",
        "04_inference/detection_rates_participant_bootstrap.csv",
        "RR<=20%, reduced, CovIC, post-only",
        "Grouped bar plot and detection-rate table",
    )

    effects = read_csv(root / "04_inference/real_vs_pseudo_effect_sizes.csv")
    effects = to_numeric(
        effects,
        [
            "WindowLength_sec",
            "RRThreshold",
            "RealRate",
            "PseudoRate",
            "RiskDifference",
            "RiskDifferenceCI_Lower",
            "RiskDifferenceCI_Upper",
            "RiskRatio",
            "RiskRatioCI_Lower",
            "RiskRatioCI_Upper",
            "OddsRatio",
            "OddsRatioCI_Lower",
            "OddsRatioCI_Upper",
        ],
    )
    primary_effects = effects[
        (effects["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (effects["Representation"] == PRIMARY_REPRESENTATION)
        & (effects["SearchWindow"] == PRIMARY_SEARCH)
        & (effects["Method"] == PRIMARY_METHOD)
    ].copy()
    perm = read_csv(root / "04_inference/participant_sign_flip_tests.csv")
    perm = to_numeric(perm, ["WindowLength_sec", "RRThreshold", "PermutationPValue"])
    perm = perm[
        (perm["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (perm["Representation"] == PRIMARY_REPRESENTATION)
        & (perm["SearchWindow"] == PRIMARY_SEARCH)
        & (perm["Method"] == PRIMARY_METHOD)
        & (perm["Outcome"] == "Detection rate")
    ][["WindowLength_sec", "TransitionType", "PseudoComparison", "PermutationPValue"]]
    primary_effects = primary_effects.merge(
        perm,
        on=["WindowLength_sec", "TransitionType", "PseudoComparison"],
        how="left",
    )
    write_csv(primary_effects, out / "main/figure_2_risk_differences.csv")
    write_csv(primary_effects, out / "tables/table_3_real_pseudo_effects.csv")
    record(
        "main/figure_2_risk_differences.csv",
        "04_inference/real_vs_pseudo_effect_sizes.csv; 04_inference/participant_sign_flip_tests.csv",
        "Primary filters; detection-rate permutation P value merged",
        "Risk-difference forest plot/table",
    )

    # ------------------------------------------------------------------
    # Population timing and multiplicity
    # ------------------------------------------------------------------
    timing = read_csv(root / "04_inference/population_shared_timing_summary.csv")
    timing = to_numeric(
        timing,
        [
            "WindowLength_sec",
            "RRThreshold",
            "CandidateTime_sec",
            "WindowSupportStart_sec",
            "WindowSupportEnd_sec",
            "PeakScore",
            "EmpiricalPValue_PseudoA",
            "EmpiricalPValue_PseudoNA",
            "BootstrapExceedsDualThresholdFraction",
            "CandidateTimeCI95_Lower_sec",
            "CandidateTimeCI95_Upper_sec",
        ],
    )
    timing_main = timing[
        (timing["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (timing["Endpoint"] == TIMING_ENDPOINT)
        & (timing["SearchWindow"] == PRIMARY_SEARCH)
        & timing["Representation"].isin(TIMING_REPRESENTATIONS)
    ].copy()
    timing_main["TimingDualPValue"] = timing_main[
        ["EmpiricalPValue_PseudoA", "EmpiricalPValue_PseudoNA"]
    ].max(axis=1)
    timing_main["TimingDualPValue_Holm12"] = holm_adjust(timing_main["TimingDualPValue"])
    timing_main["NominalDualControlSupport"] = timing_main["TimingDualPValue"] < 0.05
    timing_main["MultiplicityAdjustedSupport"] = timing_main["TimingDualPValue_Holm12"] < 0.05
    timing_main["WindowSupport"] = timing_main.apply(
        lambda r: f"{r['WindowSupportStart_sec']:g} to {r['WindowSupportEnd_sec']:g} s", axis=1
    )
    write_csv(timing_main, out / "main/figure_3_timing_summary.csv")
    write_csv(timing_main, out / "tables/table_4_population_timing.csv")
    record(
        "main/figure_3_timing_summary.csv",
        "04_inference/population_shared_timing_summary.csv",
        "RR<=20%, post-only, departure magnitude, reduced/PCA; dual P=max(Pseudo A, Pseudo NA); Holm across 12 tests",
        "Timing figure annotations and Table 4",
    )

    profiles = read_csv(root / "04_inference/population_shared_timing_profiles.csv")
    profiles = to_numeric(
        profiles,
        [
            "WindowLength_sec",
            "RRThreshold",
            "CandidateTime_sec",
            "Score",
            "PseudoA_PointwiseMedian",
            "PseudoA_PointwiseQ95",
            "PseudoNA_PointwiseMedian",
            "PseudoNA_PointwiseQ95",
        ],
    )
    profiles_main = profiles[
        (profiles["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (profiles["Endpoint"] == TIMING_ENDPOINT)
        & (profiles["SearchWindow"] == PRIMARY_SEARCH)
        & profiles["Representation"].isin(TIMING_REPRESENTATIONS)
    ].copy()
    write_csv(profiles_main, out / "main/figure_3_timing_profiles.csv")

    pseudo_null = read_csv(root / "04_inference/population_shared_timing_pseudo_null.csv.gz")
    pseudo_null = to_numeric(pseudo_null, ["WindowLength_sec", "RRThreshold", "PeakScore", "CandidateTime_sec"])
    pseudo_main = pseudo_null[
        (pseudo_null["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (pseudo_null["Endpoint"] == TIMING_ENDPOINT)
        & (pseudo_null["SearchWindow"] == PRIMARY_SEARCH)
        & pseudo_null["Representation"].isin(TIMING_REPRESENTATIONS)
    ].copy()
    write_csv(pseudo_main, out / "main/figure_3_timing_pseudo_null.csv.gz", gzip=True)

    timing_boot = read_csv(root / "04_inference/population_shared_timing_bootstrap.csv.gz")
    timing_boot = to_numeric(timing_boot, ["WindowLength_sec", "RRThreshold", "CandidateTime_sec", "PeakScore"])
    boot_main = timing_boot[
        (timing_boot["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (timing_boot["Endpoint"] == TIMING_ENDPOINT)
        & (timing_boot["SearchWindow"] == PRIMARY_SEARCH)
        & timing_boot["Representation"].isin(TIMING_REPRESENTATIONS)
    ].copy()
    write_csv(boot_main, out / "main/figure_3_timing_bootstrap.csv.gz", gzip=True)
    record(
        "main/figure_3_timing_profiles.csv; main/figure_3_timing_pseudo_null.csv.gz; main/figure_3_timing_bootstrap.csv.gz",
        "04_inference/population_shared_timing_profiles.csv; pseudo_null.csv.gz; bootstrap.csv.gz",
        "Post-only departure-magnitude timing, RR<=20%, reduced/PCA",
        "Timing line, null-distribution and bootstrap panels",
    )

    # ------------------------------------------------------------------
    # Method heat map
    # ------------------------------------------------------------------
    method_det = detection[
        (detection["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (detection["Representation"] == PRIMARY_REPRESENTATION)
        & (detection["SearchWindow"] == PRIMARY_SEARCH)
        & detection["Method"].isin(METHOD_ORDER)
    ].copy()
    rows: list[dict[str, object]] = []
    for (method, window, direction_name), group in method_det.groupby(
        ["Method", "WindowLength_sec", "TransitionType"]
    ):
        real = group[group["BoundaryKind"] == "Real"]
        pseudo_a = group[(group["BoundaryKind"] == "Pseudo") & (group["PseudoCondition"] == "A")]
        pseudo_na = group[(group["BoundaryKind"] == "Pseudo") & (group["PseudoCondition"] == "NA")]
        if real.empty or pseudo_a.empty or pseudo_na.empty:
            continue
        real_rate = float(real.iloc[0]["DetectionRate"])
        pa = float(pseudo_a.iloc[0]["DetectionRate"])
        pna = float(pseudo_na.iloc[0]["DetectionRate"])
        rows.append(
            {
                "Method": method,
                "MethodLabel": METHOD_LABELS[method],
                "MethodOrder": METHOD_ORDER.index(method),
                "WindowLength_sec": int(window),
                "TransitionType": direction_name,
                "RealRate": real_rate,
                "PseudoARate": pa,
                "PseudoNARate": pna,
                "MaxPseudoRate": max(pa, pna),
                "SpecificityMargin": real_rate - max(pa, pna),
            }
        )
    method_heatmap = pd.DataFrame(rows).sort_values(["MethodOrder", "WindowLength_sec", "TransitionType"])
    write_csv(method_heatmap, out / "main/figure_4_method_specificity_heatmap.csv")
    write_csv(method_heatmap, out / "tables/table_s1_method_comparison.csv")
    record(
        "main/figure_4_method_specificity_heatmap.csv",
        "04_inference/detection_rates_participant_bootstrap.csv",
        "RR<=20%, reduced, post-only; real rate minus the larger pseudo rate",
        "Method-comparison heat map",
    )

    # ------------------------------------------------------------------
    # Feature-level heat map with manuscript-wide FDR
    # ------------------------------------------------------------------
    feature = read_csv(root / "04_inference/feature_direction_GEE.csv")
    feature = to_numeric(
        feature,
        ["WindowLength_sec", "RRThreshold", "Estimate", "StdError", "CI95_Lower", "CI95_Upper", "PValue"],
    )
    feature_main = feature[
        (feature["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (feature["Term"] == "C(TransitionType)[T.A_to_NA]")
    ].copy()
    feature_main["QValue_Global69"] = bh_adjust(feature_main["PValue"])
    feature_main["SupportedGlobalFDR"] = feature_main["QValue_Global69"] < 0.05
    write_csv(feature_main, out / "supplement/figure_s1_feature_direction_heatmap.csv")
    write_csv(feature_main, out / "tables/table_s2_feature_direction_global_fdr.csv")
    record(
        "supplement/figure_s1_feature_direction_heatmap.csv",
        "04_inference/feature_direction_GEE.csv",
        "RR<=20%, transition-direction term; BH across all 69 feature-by-window tests",
        "Feature direction heat map and supplementary table",
    )

    # ------------------------------------------------------------------
    # Boundary order, legacy false positives and LOPO
    # ------------------------------------------------------------------
    boundary_order = to_numeric(
        read_csv(root / "05_sensitivity/boundary_order_sequence_position.csv"),
        [
            "WindowLength_sec",
            "RRThreshold",
            "BoundaryOrder",
            "DetectionRate",
            "MeanMagnitude",
            "DetectionCI95_Lower",
            "DetectionCI95_Upper",
        ],
    )
    boundary_order = boundary_order[boundary_order["RRThreshold"] == PRIMARY_RR_THRESHOLD].copy()
    write_csv(boundary_order, out / "supplement/figure_s2_boundary_order_heatmap.csv")
    write_csv(boundary_order, out / "tables/table_s3_boundary_order.csv")

    legacy = to_numeric(
        read_csv(root / "02_calibration/legacy_fixed6_simulated_null_fpr.csv"),
        ["WindowLength_sec", "NullSimulations", "Fixed6FalsePositiveRate"],
    )
    legacy = legacy[(legacy["Representation"] == PRIMARY_REPRESENTATION) & (legacy["SearchWindow"] == PRIMARY_SEARCH)].copy()
    write_csv(legacy, out / "supplement/figure_s3_legacy_null_fpr.csv")
    write_csv(legacy, out / "tables/table_s4_legacy_null_fpr.csv")

    lopo = to_numeric(
        read_csv(root / "05_sensitivity/population_shared_timing_lopo.csv"),
        ["WindowLength_sec", "RRThreshold", "CandidateTime_sec", "PeakScore"],
    )
    lopo = lopo[
        (lopo["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (lopo["Endpoint"] == TIMING_ENDPOINT)
        & (lopo["SearchWindow"] == PRIMARY_SEARCH)
        & lopo["Representation"].isin(TIMING_REPRESENTATIONS)
    ].copy()
    write_csv(lopo, out / "supplement/figure_s4_timing_lopo.csv")
    lopo_summary = (
        lopo.groupby(["WindowLength_sec", "Representation", "TransitionType"], as_index=False)
        .agg(
            OmittedParticipants=("OmittedSubject", "nunique"),
            MedianCandidateTime_sec=("CandidateTime_sec", "median"),
            MinimumCandidateTime_sec=("CandidateTime_sec", "min"),
            MaximumCandidateTime_sec=("CandidateTime_sec", "max"),
            MedianPeakScore=("PeakScore", "median"),
        )
    )
    write_csv(lopo_summary, out / "tables/table_s5_timing_lopo_summary.csv")

    power = to_numeric(
        read_csv(root / "02_calibration/population_shared_timing_simulation_power.csv"),
        [
            "WindowLength_sec",
            "RRThreshold",
            "EffectSizeSD",
            "AffectedFraction",
            "TrueTime_sec",
            "DetectionPower",
            "MedianEstimatedTime_sec",
            "MedianAbsoluteTimingError_sec",
            "MeanAbsoluteTimingError_sec",
        ],
    )
    power = power[
        (power["WindowLength_sec"] == 45)
        & (power["RRThreshold"] == PRIMARY_RR_THRESHOLD)
        & (power["Endpoint"] == TIMING_ENDPOINT)
        & (power["TransitionType"] == "A_to_NA")
        & (power["SearchWindow"] == PRIMARY_SEARCH)
        & (power["AffectedFraction"] == 0.5)
        & power["Representation"].isin(TIMING_REPRESENTATIONS)
    ].copy()
    write_csv(power, out / "supplement/figure_s5_timing_power.csv")
    write_csv(power, out / "tables/table_s6_timing_power.csv")

    # Exact condition-specific quality table for supplement.
    write_csv(quality, out / "tables/table_s7_quality_by_condition.csv")

    # Source manifest and analysis metadata.
    manifest_df = pd.DataFrame(manifest)
    write_csv(manifest_df, out / "source_manifest.csv")
    metadata = {
        "primary_method": PRIMARY_METHOD,
        "primary_representation": PRIMARY_REPRESENTATION,
        "primary_rr_threshold": PRIMARY_RR_THRESHOLD,
        "primary_search": PRIMARY_SEARCH,
        "timing_endpoint": TIMING_ENDPOINT,
        "timing_representations": list(TIMING_REPRESENTATIONS),
        "notes": [
            "NA is read as a literal condition code, not a missing-value marker.",
            "Magnitude P values use Holm correction across the three window tests for each outcome.",
            "Timing dual P is max(Pseudo-A P, Pseudo-NA P), followed by Holm correction across 12 post-only departure-magnitude tests.",
            "Feature direction Q values use BH correction across all 69 feature-by-window tests.",
        ],
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if temp is not None:
        temp.cleanup()
    print(f"Prepared manuscript data in: {out}")


if __name__ == "__main__":
    main()
