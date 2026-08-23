from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .boundaries import (
    build_pseudo_candidate_table,
    construct_pseudo_boundary_observations,
    construct_real_boundary_observations,
    match_pseudo_controls,
)
from .calibration import expand_pseudo_results
from .config import RunConfig
from .detectors import pelt_grid_boundary_collection, score_boundary_collection
from .normalization import SymmetricNormalizationStore, crossfit_stable_normalization
from .representations import derive_reduced_feature_set, fit_representation_specs
from .utils import ensure_dir, quantile_higher


def _apply_training_thresholds(
    training_pseudo: pd.DataFrame,
    heldout_real: pd.DataFrame,
    heldout_pseudo: pd.DataFrame,
    methods: Sequence[str],
    target_fpr: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow",
        "Method", "TransitionType",
    ]
    rows: list[pd.DataFrame] = []
    thresholds: list[dict[str, object]] = []
    for group_key, train_raw in training_pseudo.groupby(
        group_columns, dropna=False, sort=False
    ):
        key = dict(zip(group_columns, group_key))
        if key["Method"] not in methods:
            continue
        train = (
            train_raw.sort_values("MatchScore")
            .groupby(["CandidateBlockID", "PseudoCondition"] + group_columns, as_index=False)
            .first()
        )
        threshold = quantile_higher(train["Score"], 1.0 - target_fpr)
        thresholds.append(
            {
                **key,
                "Threshold": threshold,
                "TrainingParticipants": train["Subject"].nunique(),
                "TrainingPseudoBlocks": train["CandidateBlockID"].nunique(),
                "TrainingFPR": float(train["Score"].ge(threshold).mean()),
            }
        )
        for source in [heldout_real, heldout_pseudo]:
            group = source.copy()
            for column, value in key.items():
                group = group[group[column].eq(value)]
            if group.empty:
                continue
            group = group.copy()
            group["SourceMethod"] = key["Method"]
            group["Method"] = f"{key['Method']}_end_to_end_LOPO"
            group["Threshold"] = threshold
            group["Detected"] = group["Score"].ge(threshold)
            group["Latency_sec"] = np.where(
                group["Detected"], group["CandidateLatency_sec"], np.nan
            )
            group["CalibrationScheme"] = "other_18_participants_pseudo_threshold"
            rows.append(group)
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        pd.DataFrame(thresholds),
    )


def _choose_multiplier(training: pd.DataFrame, target_fpr: float) -> tuple[float, float]:
    summary = (
        training.groupby("PenaltyMultiplier", as_index=False)
        .agg(FPR=("DetectedAtPenalty", "mean"), Blocks=("CandidateBlockID", "nunique"))
    )
    acceptable = summary[summary["FPR"].le(target_fpr)].copy()
    if acceptable.empty:
        chosen = summary.sort_values("PenaltyMultiplier").iloc[-1]
    else:
        acceptable["Distance"] = (acceptable["FPR"] - target_fpr).abs()
        chosen = acceptable.sort_values(["Distance", "PenaltyMultiplier"]).iloc[0]
    return float(chosen.PenaltyMultiplier), float(chosen.FPR)


def _apply_training_pelt(
    training_pseudo: pd.DataFrame,
    heldout_real: pd.DataFrame,
    heldout_pseudo: pd.DataFrame,
    target_fpr: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow",
        "CostModel", "TransitionType",
    ]
    rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    for group_key, train_raw in training_pseudo.groupby(
        group_columns, dropna=False, sort=False
    ):
        key = dict(zip(group_columns, group_key))
        if str(key["CostModel"]) != "l2":
            continue
        train = (
            train_raw.sort_values("MatchScore")
            .groupby(
                ["CandidateBlockID", "PseudoCondition", "PenaltyMultiplier"]
                + group_columns,
                as_index=False,
            )
            .first()
        )
        multiplier, fpr = _choose_multiplier(train, target_fpr)
        calibration_rows.append(
            {
                **key,
                "SelectedPenaltyMultiplier": multiplier,
                "TrainingFPR": fpr,
                "TrainingPseudoBlocks": train["CandidateBlockID"].nunique(),
            }
        )
        for source in [heldout_real, heldout_pseudo]:
            group = source.copy()
            for column, value in key.items():
                group = group[group[column].eq(value)]
            group = group[np.isclose(group["PenaltyMultiplier"], multiplier)].copy()
            if group.empty:
                continue
            group["Method"] = "PELT_L2_end_to_end_LOPO"
            group["Detected"] = group["DetectedAtPenalty"].astype(bool)
            group["Latency_sec"] = np.where(
                group["Detected"], group["CandidateLatency_sec"], np.nan
            )
            group["Threshold"] = group["Penalty"]
            group["CalibrationScheme"] = "other_18_participants_pseudo_penalty"
            rows.append(group)
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        pd.DataFrame(calibration_rows),
    )


def run_end_to_end_lopo(
    data: pd.DataFrame,
    real_inventory: pd.DataFrame,
    config: RunConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit feature selection, PCA/covariance and thresholds for each held-out subject.

    The held-out participant contributes neither to reduced-feature selection,
    PCA/covariance estimation nor empirical pseudo-boundary calibration. Their
    own stable sessions are used only for unsupervised within-participant scaling,
    with the analysed boundary session(s) excluded.
    """
    ensure_dir(output_dir)
    detailed_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    fold_audit_rows: list[dict[str, object]] = []
    subjects = sorted(data["Subject"].astype(str).unique())

    for window_sec in config.windows:
        window_data = data[
            data["WindowLength_sec"].eq(window_sec)
            & data["StepSize_sec"].eq(config.step_sec)
        ].copy()
        for rr_threshold in config.lopo_rr_thresholds:
            normalizer = SymmetricNormalizationStore(
                window_data,
                rr_threshold,
                config.stable_edge_sec,
                config.seed + window_sec + int(rr_threshold),
            )
            stable_z = crossfit_stable_normalization(normalizer)
            real_obs, real_eligibility = construct_real_boundary_observations(
                window_data,
                real_inventory,
                normalizer,
                rr_threshold,
                config.pre_sec,
                config.post_sec,
                config.min_segment_points,
            )
            candidates = build_pseudo_candidate_table(
                window_data,
                rr_threshold,
                config.pre_sec,
                config.post_sec,
                config.min_segment_points,
                config.pseudo_block_separation_sec,
            )
            matches, _ = match_pseudo_controls(
                real_inventory,
                real_eligibility,
                candidates,
                min(config.pseudo_controls_per_boundary, 20),
                config.pseudo_crossfit_folds,
                config.seed + window_sec * 31 + int(rr_threshold),
            )
            pseudo_obs, _ = construct_pseudo_boundary_observations(
                window_data,
                matches,
                candidates,
                normalizer,
                rr_threshold,
                config.pre_sec,
                config.post_sec,
                config.min_segment_points,
            )

            for heldout_index, heldout in enumerate(subjects, start=1):
                logging.info(
                    "End-to-end LOPO %s/%s: held out %s, window=%ss, RR<=%s%%",
                    heldout_index, len(subjects), heldout, window_sec, rr_threshold,
                )
                training_stable = stable_z[~stable_z["Subject"].astype(str).eq(heldout)].copy()
                reduced, _, _ = derive_reduced_feature_set(
                    training_stable, config.reduced_abs_correlation, output_dir=None
                )
                specs, _, _, audit = fit_representation_specs(
                    training_stable,
                    window_sec,
                    config.step_sec,
                    reduced,
                    config.pca_variance_target,
                    config.seed + stable_hash_subject(heldout),
                    output_dir=None,
                )
                specs = {name: spec for name, spec in specs.items() if name in {"reduced", "independent_pca"}}
                fold_audit_rows.append(
                    {
                        "HeldoutSubject": heldout,
                        "WindowLength_sec": window_sec,
                        "RRThreshold": rr_threshold,
                        "TrainingParticipants": len(subjects) - 1,
                        "ReducedFeatures": ";".join(reduced),
                        "ReducedFeatureCount": len(reduced),
                        "Representations": ";".join(specs),
                    }
                )

                train_pseudo_obs = pseudo_obs[~pseudo_obs["Subject"].astype(str).eq(heldout)]
                held_pseudo_obs = pseudo_obs[pseudo_obs["Subject"].astype(str).eq(heldout)]
                held_real_obs = real_obs[real_obs["Subject"].astype(str).eq(heldout)]
                train_matches = matches[~matches["Subject"].astype(str).eq(heldout)]
                held_matches = matches[matches["Subject"].astype(str).eq(heldout)]

                train_pseudo_score = score_boundary_collection(
                    train_pseudo_obs,
                    specs,
                    window_sec,
                    rr_threshold,
                    config.min_segment_points,
                )
                held_real_score = score_boundary_collection(
                    held_real_obs,
                    specs,
                    window_sec,
                    rr_threshold,
                    config.min_segment_points,
                )
                held_pseudo_score = score_boundary_collection(
                    held_pseudo_obs,
                    specs,
                    window_sec,
                    rr_threshold,
                    config.min_segment_points,
                )
                train_expanded = expand_pseudo_results(train_pseudo_score, train_matches)
                held_expanded = expand_pseudo_results(held_pseudo_score, held_matches)
                if not held_real_score.empty:
                    held_real_score["MatchedRealBoundaryID"] = held_real_score["BoundaryID"]
                    held_real_score["HeldoutSubject"] = heldout
                if not held_expanded.empty:
                    held_expanded["HeldoutSubject"] = heldout

                requested_score_methods = [
                    method for method in config.lopo_methods if method != "PELT_L2"
                ]
                score_eval, score_thresholds = _apply_training_thresholds(
                    train_expanded,
                    held_real_score,
                    held_expanded,
                    requested_score_methods,
                    config.calibration_fpr,
                )
                if not score_eval.empty:
                    score_eval["HeldoutSubject"] = heldout
                    detailed_rows.append(score_eval)
                if not score_thresholds.empty:
                    score_thresholds["HeldoutSubject"] = heldout
                    threshold_rows.append(score_thresholds)

                if "PELT_L2" in config.lopo_methods:
                    train_grid = pelt_grid_boundary_collection(
                        train_pseudo_obs,
                        specs,
                        window_sec,
                        rr_threshold,
                        config.min_segment_points,
                        config.pelt_multipliers,
                        config.require_ruptures,
                        representations=("reduced",),
                    )
                    held_real_grid = pelt_grid_boundary_collection(
                        held_real_obs,
                        specs,
                        window_sec,
                        rr_threshold,
                        config.min_segment_points,
                        config.pelt_multipliers,
                        config.require_ruptures,
                        representations=("reduced",),
                    )
                    held_pseudo_grid = pelt_grid_boundary_collection(
                        held_pseudo_obs,
                        specs,
                        window_sec,
                        rr_threshold,
                        config.min_segment_points,
                        config.pelt_multipliers,
                        config.require_ruptures,
                        representations=("reduced",),
                    )
                    train_grid_expanded = expand_pseudo_results(train_grid, train_matches)
                    held_grid_expanded = expand_pseudo_results(held_pseudo_grid, held_matches)
                    if not held_real_grid.empty:
                        held_real_grid["MatchedRealBoundaryID"] = held_real_grid["BoundaryID"]
                    pelt_eval, pelt_thresholds = _apply_training_pelt(
                        train_grid_expanded,
                        held_real_grid,
                        held_grid_expanded,
                        config.calibration_fpr,
                    )
                    if not pelt_eval.empty:
                        pelt_eval["HeldoutSubject"] = heldout
                        detailed_rows.append(pelt_eval)
                    if not pelt_thresholds.empty:
                        pelt_thresholds["HeldoutSubject"] = heldout
                        threshold_rows.append(pelt_thresholds)

                # The next iteration overwrites all held-out-specific tables; an
                # explicit collection pass keeps long 19-fold runs memory-stable.
                gc.collect()

    detail = pd.concat(detailed_rows, ignore_index=True) if detailed_rows else pd.DataFrame()
    thresholds = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    audit = pd.DataFrame(fold_audit_rows)
    if not detail.empty:
        detail.to_csv(output_dir / "end_to_end_lopo_detailed.csv.gz", index=False, compression="gzip")
        summary = (
            detail.groupby(
                [
                    "WindowLength_sec", "RRThreshold", "Representation", "SearchWindow",
                    "Method", "BoundaryKind", "TransitionType", "PseudoCondition",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                HeldoutParticipants=("HeldoutSubject", "nunique"),
                BoundariesOrMatches=("BoundaryID", "size"),
                Detections=("Detected", "sum"),
                DetectionRate=("Detected", "mean"),
                MedianLatencyDetected_sec=("Latency_sec", "median"),
            )
        )
    else:
        summary = pd.DataFrame()
    thresholds.to_csv(output_dir / "end_to_end_lopo_training_thresholds.csv", index=False)
    audit.to_csv(output_dir / "end_to_end_lopo_fold_audit.csv", index=False)
    summary.to_csv(output_dir / "end_to_end_lopo_summary.csv", index=False)
    return detail, summary, audit


def stable_hash_subject(subject: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(subject))
