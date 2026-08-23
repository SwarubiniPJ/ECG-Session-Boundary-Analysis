from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .utils import benjamini_hochberg, quantile_higher


def expand_pseudo_results(
    pseudo_results: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    if pseudo_results.empty or matches.empty:
        return pd.DataFrame()
    match_columns = [
        "PseudoMatchID", "MatchedRealBoundaryID", "Subject", "TransitionType",
        "TransitionLabel", "BoundaryOrder", "PseudoCondition", "CandidateID",
        "CandidateBlockID", "PseudoFold", "MatchScore", "MatchRank",
    ]
    expanded = matches[match_columns].merge(
        pseudo_results,
        on=["CandidateID", "CandidateBlockID", "PseudoFold", "Subject", "PseudoCondition"],
        how="inner",
        suffixes=("", "_score"),
        validate="many_to_many",
    )
    expanded["BoundaryKind"] = "Pseudo"
    return expanded


def _crossfit_thresholds_for_group(
    pseudo: pd.DataFrame,
    target_fpr: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    folds = sorted(int(value) for value in pseudo["PseudoFold"].dropna().unique() if int(value) >= 0)
    for fold in folds:
        calibration = pseudo[pseudo["PseudoFold"].ne(fold)]
        heldout = pseudo[pseudo["PseudoFold"].eq(fold)]
        threshold = quantile_higher(calibration["Score"], 1.0 - target_fpr)
        rows.append(
            {
                "CalibrationFold": fold,
                "Threshold": threshold,
                "CalibrationBlocks": calibration["CandidateBlockID"].nunique(),
                "HeldoutBlocks": heldout["CandidateBlockID"].nunique(),
                "HeldoutFPR": float(heldout["Score"].ge(threshold).mean()) if len(heldout) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def crossfit_score_calibration(
    real_scores: pd.DataFrame,
    pseudo_scores_expanded: pd.DataFrame,
    target_fpr: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direction-specific cross-fitted calibration on disjoint pseudo blocks."""
    if real_scores.empty or pseudo_scores_expanded.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_columns = [
        "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow",
        "Method", "TransitionType",
    ]
    threshold_rows: list[pd.DataFrame] = []
    evaluation_rows: list[pd.DataFrame] = []

    for group_key, pseudo_group_raw in pseudo_scores_expanded.groupby(
        group_columns, dropna=False, sort=False
    ):
        key = dict(zip(group_columns, group_key))
        # One score per independent pseudo block and condition prevents dense or
        # repeated matches from receiving excessive calibration weight.
        pseudo_group = (
            pseudo_group_raw.sort_values("MatchScore")
            .groupby(
                ["CandidateBlockID", "PseudoCondition", "PseudoFold"] + group_columns,
                as_index=False,
                dropna=False,
            )
            .first()
        )
        fold_table = _crossfit_thresholds_for_group(pseudo_group, target_fpr)
        if fold_table.empty:
            continue
        for column, value in key.items():
            fold_table[column] = value
        fold_table["TargetFPR"] = target_fpr
        fold_table["CalibrationStratification"] = "transition_direction_specific"
        threshold_rows.append(fold_table)

        final_threshold = float(np.nanmedian(fold_table["Threshold"]))
        real_group = real_scores.copy()
        for column, value in key.items():
            real_group = real_group[real_group[column].eq(value)]
        if not real_group.empty:
            real_eval = real_group.copy()
            calibrated_method = (
                "LegacyBIC_crossfit" if key["Method"] == "LegacyBIC_fixed6"
                else f"{key['Method']}_crossfit"
            )
            real_eval["SourceMethod"] = key["Method"]
            real_eval["Method"] = calibrated_method
            real_eval["Threshold"] = final_threshold
            real_eval["Detected"] = real_eval["Score"].ge(final_threshold)
            real_eval["Latency_sec"] = np.where(
                real_eval["Detected"], real_eval["CandidateLatency_sec"], np.nan
            )
            real_eval["CalibrationFold"] = -1
            real_eval["CalibrationScheme"] = "median_of_direction_specific_crossfit_thresholds"
            evaluation_rows.append(real_eval)

        calibrated_method = (
            "LegacyBIC_crossfit" if key["Method"] == "LegacyBIC_fixed6"
            else f"{key['Method']}_crossfit"
        )
        for threshold_row in fold_table.itertuples(index=False):
            fold = int(threshold_row.CalibrationFold)
            threshold = float(threshold_row.Threshold)
            heldout = pseudo_group_raw[pseudo_group_raw["PseudoFold"].eq(fold)].copy()
            if heldout.empty:
                continue
            heldout["SourceMethod"] = key["Method"]
            heldout["Method"] = calibrated_method
            heldout["Threshold"] = threshold
            heldout["Detected"] = heldout["Score"].ge(threshold)
            heldout["Latency_sec"] = np.where(
                heldout["Detected"], heldout["CandidateLatency_sec"], np.nan
            )
            heldout["CalibrationFold"] = fold
            heldout["CalibrationScheme"] = "heldout_disjoint_pseudo_block"
            evaluation_rows.append(heldout)

    # Preserve the original fixed threshold as a diagnostic legacy method.
    legacy_real = real_scores[real_scores["Method"].eq("LegacyBIC_fixed6")].copy()
    legacy_pseudo = pseudo_scores_expanded[
        pseudo_scores_expanded["Method"].eq("LegacyBIC_fixed6")
    ].copy()
    for frame in [legacy_real, legacy_pseudo]:
        if frame.empty:
            continue
        frame["Threshold"] = 6.0
        frame["Detected"] = frame["Score"].ge(6.0)
        frame["Latency_sec"] = np.where(
            frame["Detected"], frame["CandidateLatency_sec"], np.nan
        )
        frame["CalibrationFold"] = -1
        frame["CalibrationScheme"] = "legacy_fixed_delta_BIC_6"
        evaluation_rows.append(frame)

    return (
        pd.concat(evaluation_rows, ignore_index=True) if evaluation_rows else pd.DataFrame(),
        pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame(),
    )


def _choose_pelt_multiplier(
    calibration: pd.DataFrame,
    multipliers: Sequence[float],
    target_fpr: float,
) -> tuple[float, float, int]:
    summary = (
        calibration.groupby("PenaltyMultiplier", as_index=False)
        .agg(
            FalsePositiveRate=("DetectedAtPenalty", "mean"),
            Blocks=("CandidateBlockID", "nunique"),
        )
    )
    if summary.empty:
        return np.nan, np.nan, 0
    acceptable = summary[summary["FalsePositiveRate"].le(target_fpr)].copy()
    if acceptable.empty:
        chosen = summary.sort_values("PenaltyMultiplier").iloc[-1]
    else:
        acceptable["Distance"] = (acceptable["FalsePositiveRate"] - target_fpr).abs()
        chosen = acceptable.sort_values(["Distance", "PenaltyMultiplier"]).iloc[0]
    return (
        float(chosen["PenaltyMultiplier"]),
        float(chosen["FalsePositiveRate"]),
        int(chosen["Blocks"]),
    )


def crossfit_pelt_calibration(
    real_grid: pd.DataFrame,
    pseudo_grid_expanded: pd.DataFrame,
    target_fpr: float,
    multipliers: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if real_grid.empty or pseudo_grid_expanded.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_columns = [
        "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow",
        "CostModel", "TransitionType",
    ]
    evaluation_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []

    for group_key, pseudo_raw in pseudo_grid_expanded.groupby(
        group_columns, dropna=False, sort=False
    ):
        key = dict(zip(group_columns, group_key))
        pseudo = (
            pseudo_raw.sort_values("MatchScore")
            .groupby(
                ["CandidateBlockID", "PseudoCondition", "PseudoFold", "PenaltyMultiplier"]
                + group_columns,
                as_index=False,
                dropna=False,
            )
            .first()
        )
        fold_choices: list[float] = []
        for fold in sorted(pseudo["PseudoFold"].dropna().astype(int).unique()):
            calibration = pseudo[pseudo["PseudoFold"].ne(fold)]
            heldout = pseudo[pseudo["PseudoFold"].eq(fold)]
            multiplier, calibration_fpr, blocks = _choose_pelt_multiplier(
                calibration, multipliers, target_fpr
            )
            if not np.isfinite(multiplier):
                continue
            fold_choices.append(multiplier)
            heldout_selected = heldout[
                np.isclose(heldout["PenaltyMultiplier"], multiplier)
            ].copy()
            heldout_fpr = float(heldout_selected["DetectedAtPenalty"].mean()) if len(heldout_selected) else np.nan
            calibration_rows.append(
                {
                    **key,
                    "CalibrationFold": int(fold),
                    "SelectedPenaltyMultiplier": multiplier,
                    "CalibrationFPR": calibration_fpr,
                    "HeldoutFPR": heldout_fpr,
                    "CalibrationBlocks": blocks,
                    "HeldoutBlocks": heldout_selected["CandidateBlockID"].nunique(),
                    "TargetFPR": target_fpr,
                }
            )
            if not heldout_selected.empty:
                heldout_selected["Method"] = f"PELT_{str(key['CostModel']).upper()}_crossfit"
                heldout_selected["Detected"] = heldout_selected["DetectedAtPenalty"].astype(bool)
                heldout_selected["Latency_sec"] = np.where(
                    heldout_selected["Detected"], heldout_selected["CandidateLatency_sec"], np.nan
                )
                heldout_selected["Threshold"] = heldout_selected["Penalty"]
                heldout_selected["CalibrationFold"] = int(fold)
                heldout_selected["CalibrationScheme"] = "heldout_disjoint_pseudo_block"
                evaluation_rows.append(heldout_selected)

        if not fold_choices:
            continue
        final_multiplier = float(np.median(fold_choices))
        available = np.asarray(sorted(set(float(value) for value in multipliers)))
        final_multiplier = float(available[np.argmin(np.abs(available - final_multiplier))])
        real_group = real_grid.copy()
        for column, value in key.items():
            real_group = real_group[real_group[column].eq(value)]
        real_group = real_group[np.isclose(real_group["PenaltyMultiplier"], final_multiplier)].copy()
        if not real_group.empty:
            real_group["Method"] = f"PELT_{str(key['CostModel']).upper()}_crossfit"
            real_group["Detected"] = real_group["DetectedAtPenalty"].astype(bool)
            real_group["Latency_sec"] = np.where(
                real_group["Detected"], real_group["CandidateLatency_sec"], np.nan
            )
            real_group["Threshold"] = real_group["Penalty"]
            real_group["CalibrationFold"] = -1
            real_group["CalibrationScheme"] = "median_of_direction_specific_crossfit_multipliers"
            evaluation_rows.append(real_group)

    return (
        pd.concat(evaluation_rows, ignore_index=True) if evaluation_rows else pd.DataFrame(),
        pd.DataFrame(calibration_rows),
    )


def matched_empirical_pvalues(
    real_scores: pd.DataFrame,
    pseudo_scores_expanded: pd.DataFrame,
) -> pd.DataFrame:
    """Boundary-level empirical P values against participant-matched pseudo blocks."""
    if real_scores.empty or pseudo_scores_expanded.empty:
        return pd.DataFrame()
    key_columns = [
        "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow",
        "Method", "TransitionType",
    ]

    # Pre-aggregate once instead of scanning the complete pseudo table for every
    # real boundary and method. One candidate per independent block is retained.
    pseudo_unique = (
        pseudo_scores_expanded.sort_values("MatchScore")
        .groupby(
            key_columns
            + ["MatchedRealBoundaryID", "PseudoCondition", "CandidateBlockID"],
            as_index=False,
            dropna=False,
        )
        .first()
    )
    lookup: dict[tuple[object, ...], np.ndarray] = {}
    for lookup_key, group in pseudo_unique.groupby(
        key_columns + ["MatchedRealBoundaryID", "PseudoCondition"],
        dropna=False,
        sort=False,
    ):
        scores = group["Score"].to_numpy(dtype=float)
        lookup[tuple(lookup_key)] = scores[np.isfinite(scores)]

    rows: list[dict[str, object]] = []
    for real in real_scores.itertuples(index=False):
        base_key = tuple(getattr(real, column) for column in key_columns)
        for pseudo_condition in ["A", "NA"]:
            scores = lookup.get(base_key + (real.BoundaryID, pseudo_condition), np.empty(0))
            real_score = float(real.Score)
            pvalue = (
                float((1 + np.sum(scores >= real_score)) / (len(scores) + 1))
                if np.isfinite(real_score) and len(scores)
                else np.nan
            )
            rows.append(
                {
                    "BoundaryID": real.BoundaryID,
                    "Subject": real.Subject,
                    "BoundaryOrder": real.BoundaryOrder,
                    "TransitionType": real.TransitionType,
                    "WindowLength_sec": real.WindowLength_sec,
                    "RRThreshold": real.RRThreshold,
                    "Representation": real.Representation,
                    "SearchWindow": real.SearchWindow,
                    "Method": real.Method,
                    "PseudoCondition": pseudo_condition,
                    "RealScore": real_score,
                    "PseudoControls": int(len(scores)),
                    "PseudoMedianScore": float(np.median(scores)) if len(scores) else np.nan,
                    "PseudoQ95Score": quantile_higher(scores, 0.95) if len(scores) else np.nan,
                    "RealMinusPseudoMedian": (
                        real_score - float(np.median(scores)) if len(scores) else np.nan
                    ),
                    "EmpiricalPValue": pvalue,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["EmpiricalQValue_BH"] = result.groupby(
        key_columns + ["PseudoCondition"], group_keys=False
    )["EmpiricalPValue"].transform(benjamini_hochberg)
    return result
