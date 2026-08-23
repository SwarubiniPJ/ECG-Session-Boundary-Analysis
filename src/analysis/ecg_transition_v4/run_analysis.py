from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .boundaries import (
    build_pseudo_candidate_table,
    construct_pseudo_boundary_observations,
    construct_real_boundary_observations,
    match_pseudo_controls,
)
from .calibration import (
    crossfit_pelt_calibration,
    crossfit_score_calibration,
    expand_pseudo_results,
    matched_empirical_pvalues,
)
from .config import ALLOWED_WINDOWS, RunConfig
from .detectors import pelt_grid_boundary_collection, score_boundary_collection
from .inference import compute_feature_effects, quality_sensitivity_tables
from .io_preprocess import (
    build_real_boundary_inventory,
    configuration_audit,
    read_master,
)
from .lopo import run_end_to_end_lopo
from .population_timing import run_population_shared_timing
from .normalization import SymmetricNormalizationStore, crossfit_stable_normalization
from .representations import derive_reduced_feature_set, fit_representation_specs
from .simulation import run_simulation_calibration
from .utils import (
    combine_gzip_csv_parts,
    combine_plain_csv_parts,
    ensure_dir,
    file_sha256,
    package_versions,
    setup_logging,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nature-oriented ECG transition validation using 30-, 45-, and 60-second "
            "windows, cross-fitted pseudo controls, gradual-change methods, and "
            "end-to-end participant-held-out validation, and a participant-pooled "
            "pseudo-calibrated timing model for both transition directions."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Master ECG/HRV window CSV")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Nature_Timing_Validated_Results_V4"),
    )
    parser.add_argument("--windows", nargs="+", type=int, default=list(ALLOWED_WINDOWS))
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--rr-thresholds", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    parser.add_argument("--pseudo-controls-per-boundary", type=int, default=50)
    parser.add_argument("--pseudo-block-separation", type=float, default=120.0)
    parser.add_argument("--pseudo-folds", type=int, default=4)
    parser.add_argument(
        "--pelt-multipliers",
        nargs="+",
        type=float,
        default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--null-simulations", type=int, default=400)
    parser.add_argument("--power-simulations", type=int, default=100)

    parser.add_argument("--no-population-timing", action="store_true")
    parser.add_argument("--timing-windows", nargs="+", type=int, default=[30, 45, 60])
    parser.add_argument("--timing-rr-thresholds", nargs="+", type=float, default=[20.0])
    parser.add_argument(
        "--timing-representations",
        nargs="+",
        default=["reduced", "independent_pca"],
    )
    parser.add_argument(
        "--timing-endpoints",
        nargs="+",
        default=["departure_magnitude"],
        choices=["departure_magnitude", "signed_trajectory"],
    )
    parser.add_argument(
        "--timing-search-windows",
        nargs="+",
        default=["post_only", "anticipatory"],
        choices=["post_only", "anticipatory"],
    )
    parser.add_argument("--timing-pseudo-draws", type=int, default=1000)
    parser.add_argument("--timing-bootstrap", type=int, default=2000)
    parser.add_argument("--timing-power-simulations", type=int, default=50)
    parser.add_argument(
        "--timing-simulation-effect-sizes",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5],
    )
    parser.add_argument(
        "--timing-simulation-affected-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument("--timing-min-unique-times-per-side", type=int, default=3)
    parser.add_argument("--timing-min-participants", type=int, default=10)
    parser.add_argument("--timing-alpha", type=float, default=0.05)
    parser.add_argument("--timing-primary-window", type=int, default=30)
    parser.add_argument("--timing-primary-rr-threshold", type=float, default=20.0)
    parser.add_argument("--timing-primary-representation", default="reduced")
    parser.add_argument(
        "--timing-primary-endpoint",
        default="departure_magnitude",
        choices=["departure_magnitude", "signed_trajectory"],
    )
    parser.add_argument("--no-timing-lopo", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure-dpi", type=int, default=400)
    parser.add_argument("--no-lopo", action="store_true")
    parser.add_argument("--lopo-rr-thresholds", nargs="+", type=float, default=[20.0])
    parser.add_argument("--require-ruptures", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(
        input_csv=args.input.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        windows=tuple(dict.fromkeys(int(value) for value in args.windows)),
        step_sec=int(args.step),
        rr_thresholds=tuple(sorted(set(float(value) for value in args.rr_thresholds))),
        pseudo_controls_per_boundary=int(args.pseudo_controls_per_boundary),
        pseudo_block_separation_sec=float(args.pseudo_block_separation),
        pseudo_crossfit_folds=int(args.pseudo_folds),
        pelt_multipliers=tuple(sorted(set(float(value) for value in args.pelt_multipliers))),
        bootstrap_replicates=int(args.bootstrap),
        permutation_replicates=int(args.permutations),
        null_simulations=int(args.null_simulations),
        power_simulations=int(args.power_simulations),
        run_population_timing=not bool(args.no_population_timing),
        timing_windows=tuple(dict.fromkeys(int(value) for value in args.timing_windows)),
        timing_rr_thresholds=tuple(
            sorted(set(float(value) for value in args.timing_rr_thresholds))
        ),
        timing_representations=tuple(dict.fromkeys(args.timing_representations)),
        timing_endpoints=tuple(dict.fromkeys(args.timing_endpoints)),
        timing_search_windows=tuple(dict.fromkeys(args.timing_search_windows)),
        timing_pseudo_draws=int(args.timing_pseudo_draws),
        timing_bootstrap_replicates=int(args.timing_bootstrap),
        timing_power_simulations=int(args.timing_power_simulations),
        timing_simulation_effect_sizes=tuple(
            sorted(set(float(value) for value in args.timing_simulation_effect_sizes))
        ),
        timing_simulation_affected_fraction=float(
            args.timing_simulation_affected_fraction
        ),
        timing_min_unique_times_per_side=int(args.timing_min_unique_times_per_side),
        timing_min_participants=int(args.timing_min_participants),
        timing_alpha=float(args.timing_alpha),
        timing_primary_window=int(args.timing_primary_window),
        timing_primary_rr_threshold=float(args.timing_primary_rr_threshold),
        timing_primary_representation=str(args.timing_primary_representation),
        timing_primary_endpoint=str(args.timing_primary_endpoint),
        timing_lopo=not bool(args.no_timing_lopo),
        run_lopo=not bool(args.no_lopo),
        lopo_rr_thresholds=tuple(sorted(set(float(value) for value in args.lopo_rr_thresholds))),
        seed=int(args.seed),
        figure_dpi=int(args.figure_dpi),
        require_ruptures=bool(args.require_ruptures),
    )
    config.validate()
    return config


def _expand_pseudo_effects(
    pseudo_effects: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    if pseudo_effects.empty or matches.empty:
        return pd.DataFrame()
    match_columns = [
        "MatchedRealBoundaryID", "Subject", "TransitionType", "BoundaryOrder",
        "PseudoCondition", "CandidateID", "CandidateBlockID", "PseudoFold",
        "MatchScore", "MatchRank",
    ]
    expanded = matches[match_columns].merge(
        pseudo_effects,
        on=["Subject", "PseudoCondition", "CandidateID", "CandidateBlockID", "PseudoFold"],
        how="inner",
        suffixes=("", "_effect"),
    )
    expanded["BoundaryKind"] = "Pseudo"
    # Retain one representative match per independent pseudo block, direction,
    # condition, and feature. Repeated matches to several real boundaries are
    # needed for boundary-level detector contrasts but should not multiply the
    # same physiological pseudo effect in feature-level summaries.
    deduplication = [
        "CandidateBlockID", "PseudoCondition", "TransitionType", "Feature",
        "WindowLength_sec", "RRThreshold",
    ]
    return (
        expanded.sort_values(["MatchScore", "MatchRank"], kind="stable")
        .drop_duplicates(deduplication)
        .reset_index(drop=True)
    )


def run(config: RunConfig) -> None:
    setup_logging(config.output_root)
    logging.info("Starting Nature-oriented V4 analysis")
    logging.info("Output root: %s", config.output_root)

    audit_dir = ensure_dir(config.output_root / "00_audit")
    representation_dir = ensure_dir(config.output_root / "01_feature_representations")
    calibration_dir = ensure_dir(config.output_root / "02_calibration")
    boundary_dir = ensure_dir(config.output_root / "03_boundary_results")
    inference_dir = ensure_dir(config.output_root / "04_inference")
    sensitivity_dir = ensure_dir(config.output_root / "05_sensitivity")
    lopo_dir = ensure_dir(config.output_root / "06_end_to_end_lopo")
    figures_dir = ensure_dir(config.output_root / "07_figures")
    paper_dir = ensure_dir(config.output_root / "08_paper_tables")

    boundary_parts_dir = ensure_dir(boundary_dir / "_parts")
    inference_parts_dir = ensure_dir(inference_dir / "_parts")
    timing_parts_dir = ensure_dir(inference_dir / "_population_timing_parts")
    timing_sensitivity_parts_dir = ensure_dir(
        sensitivity_dir / "_population_timing_parts"
    )

    data = read_master(config.input_csv)
    inventory = build_real_boundary_inventory(data)
    inventory.to_csv(audit_dir / "real_boundary_inventory.csv", index=False)
    configuration_audit(data).to_csv(audit_dir / "master_configuration_audit.csv", index=False)
    manifest = {
        "script": "ecg_transition_analysis_timing_v4.py",
        "input_csv": str(config.input_csv),
        "input_sha256": file_sha256(config.input_csv),
        "output_root": str(config.output_root),
        "config": config.__dict__,
        "software_versions": package_versions(),
        "neutral_interpretation": (
            "Detected changes are associated with video-session boundaries and cannot "
            "be interpreted as anxiety-specific physiological onset or recovery."
        ),
    }
    write_json(audit_dir / "analysis_manifest.json", manifest)

    quality_window, quality_participant, quality_failures = quality_sensitivity_tables(
        data, config.windows, config.step_sec, config.rr_thresholds
    )
    quality_window.to_csv(sensitivity_dir / "rr_quality_window_summary.csv", index=False)
    quality_participant.to_csv(sensitivity_dir / "rr_quality_by_participant.csv", index=False)
    quality_failures.to_csv(sensitivity_dir / "rr_quality_exclusion_reasons.csv", index=False)

    # Freeze the primary reduced set using 30-s, RR<=20 stable cross-fitted data.
    source_data = data[
        data["WindowLength_sec"].eq(30)
        & data["StepSize_sec"].eq(config.step_sec)
    ].copy()
    source_normalizer = SymmetricNormalizationStore(
        source_data, 20.0, config.stable_edge_sec, config.seed
    )
    source_stable_z = crossfit_stable_normalization(source_normalizer)
    reduced_features, _, reduced_selection = derive_reduced_feature_set(
        source_stable_z,
        config.reduced_abs_correlation,
        representation_dir,
    )
    reduced_selection.to_csv(
        representation_dir / "primary_reduced_feature_selection.csv", index=False
    )

    evaluation_parts: list[Path] = []
    raw_real_parts: list[Path] = []
    raw_pseudo_candidate_parts: list[Path] = []
    empirical_parts: list[Path] = []
    effect_parts: list[Path] = []
    all_thresholds: list[pd.DataFrame] = []
    all_pelt_calibration: list[pd.DataFrame] = []
    all_real_eligibility: list[pd.DataFrame] = []
    all_pseudo_audits: list[pd.DataFrame] = []
    all_match_audits: list[pd.DataFrame] = []
    all_normalization_audits: list[pd.DataFrame] = []
    all_rep_audits: list[pd.DataFrame] = []
    all_sim_thresholds: list[pd.DataFrame] = []
    all_sim_legacy: list[pd.DataFrame] = []
    all_sim_pelt: list[pd.DataFrame] = []
    all_sim_power: list[pd.DataFrame] = []

    timing_summary_parts: list[Path] = []
    timing_profile_parts: list[Path] = []
    timing_null_parts: list[Path] = []
    timing_bootstrap_parts: list[Path] = []
    timing_lopo_parts: list[Path] = []
    timing_direction_parts: list[Path] = []
    timing_simulation_parts: list[Path] = []

    for window_sec in config.windows:
        window_data = data[
            data["WindowLength_sec"].eq(window_sec)
            & data["StepSize_sec"].eq(config.step_sec)
        ].copy()
        for rr_threshold in config.rr_thresholds:
            logging.info(
                "Preparing window=%ss, RR<=%s%%", window_sec, rr_threshold
            )
            normalizer = SymmetricNormalizationStore(
                window_data,
                rr_threshold,
                config.stable_edge_sec,
                config.seed + window_sec * 100 + int(rr_threshold),
            )
            stable_z = crossfit_stable_normalization(normalizer)
            specs, pca_variance, pca_loadings, rep_audit = fit_representation_specs(
                stable_z,
                window_sec,
                config.step_sec,
                reduced_features,
                config.pca_variance_target,
                config.seed + window_sec * 17 + int(rr_threshold),
                representation_dir,
            )
            rep_audit = rep_audit.copy()
            rep_audit["RRThreshold"] = rr_threshold
            all_rep_audits.append(rep_audit)

            real_obs, real_eligibility = construct_real_boundary_observations(
                window_data,
                inventory,
                normalizer,
                rr_threshold,
                config.pre_sec,
                config.post_sec,
                config.min_segment_points,
            )
            real_eligibility.insert(0, "WindowLength_sec", window_sec)
            all_real_eligibility.append(real_eligibility)

            combo_label = f"w{window_sec}_rr{rr_threshold:g}"

            # Run simulation calibration before constructing the much larger
            # matched pseudo-control tables. This keeps peak memory bounded.
            if rr_threshold == 20.0:
                time_templates = [
                    group.sort_values("RelativeCenter_sec")["RelativeCenter_sec"].to_numpy(dtype=float)
                    for _, group in real_obs.groupby("BoundaryID", sort=False)
                ]
                for representation in config.simulation_representations:
                    if representation not in specs:
                        continue
                    logging.info("Running simulations for %s, %s", combo_label, representation)
                    sim_thresholds, sim_legacy, sim_pelt, sim_power = run_simulation_calibration(
                        window_sec,
                        representation,
                        specs[representation],
                        time_templates,
                        config,
                        calibration_dir,
                    )
                    all_sim_thresholds.append(sim_thresholds)
                    all_sim_legacy.append(sim_legacy)
                    all_sim_pelt.append(sim_pelt)
                    all_sim_power.append(sim_power)

            candidates = build_pseudo_candidate_table(
                window_data,
                rr_threshold,
                config.pre_sec,
                config.post_sec,
                config.min_segment_points,
                config.pseudo_block_separation_sec,
            )
            matches, match_audit = match_pseudo_controls(
                inventory,
                real_eligibility,
                candidates,
                config.pseudo_controls_per_boundary,
                config.pseudo_crossfit_folds,
                config.seed + window_sec * 31 + int(rr_threshold),
            )
            pseudo_obs, pseudo_audit = construct_pseudo_boundary_observations(
                window_data,
                matches,
                candidates,
                normalizer,
                rr_threshold,
                config.pre_sec,
                config.post_sec,
                config.min_segment_points,
            )
            for frame in [match_audit, pseudo_audit]:
                if not frame.empty:
                    frame.insert(0, "WindowLength_sec", window_sec)
                    frame.insert(1, "RRThreshold", rr_threshold)
            all_match_audits.append(match_audit)
            all_pseudo_audits.append(pseudo_audit)
            normalizer_audit = normalizer.audit_table()
            if not normalizer_audit.empty:
                normalizer_audit.insert(0, "WindowLength_sec", window_sec)
                normalizer_audit.insert(1, "RRThreshold", rr_threshold)
                all_normalization_audits.append(normalizer_audit)

            real_obs.to_csv(
                boundary_dir / f"real_boundary_observations_{combo_label}.csv.gz",
                index=False,
                compression="gzip",
            )
            pseudo_obs.to_csv(
                boundary_dir / f"pseudo_boundary_observations_{combo_label}.csv.gz",
                index=False,
                compression="gzip",
            )
            matches.to_csv(
                boundary_dir / f"pseudo_matches_{combo_label}.csv.gz",
                index=False,
                compression="gzip",
            )

            if (
                config.run_population_timing
                and window_sec in set(config.timing_windows)
                and float(rr_threshold)
                in set(float(value) for value in config.timing_rr_thresholds)
            ):
                logging.info("Running participant-pooled timing for %s", combo_label)
                (
                    timing_summary,
                    timing_profiles,
                    timing_null,
                    timing_bootstrap,
                    timing_lopo,
                    timing_direction,
                    timing_simulation,
                ) = run_population_shared_timing(
                    real_obs,
                    pseudo_obs,
                    matches,
                    specs,
                    window_sec,
                    rr_threshold,
                    config,
                )
                timing_outputs = [
                    (
                        timing_summary,
                        timing_parts_dir / f"timing_summary_{combo_label}.csv",
                        timing_summary_parts,
                        False,
                    ),
                    (
                        timing_profiles,
                        timing_parts_dir / f"timing_profiles_{combo_label}.csv",
                        timing_profile_parts,
                        False,
                    ),
                    (
                        timing_null,
                        timing_parts_dir / f"timing_null_{combo_label}.csv.gz",
                        timing_null_parts,
                        True,
                    ),
                    (
                        timing_bootstrap,
                        timing_parts_dir / f"timing_bootstrap_{combo_label}.csv.gz",
                        timing_bootstrap_parts,
                        True,
                    ),
                    (
                        timing_lopo,
                        timing_sensitivity_parts_dir / f"timing_lopo_{combo_label}.csv",
                        timing_lopo_parts,
                        False,
                    ),
                    (
                        timing_direction,
                        timing_parts_dir / f"timing_direction_{combo_label}.csv",
                        timing_direction_parts,
                        False,
                    ),
                    (
                        timing_simulation,
                        timing_parts_dir / f"timing_simulation_{combo_label}.csv",
                        timing_simulation_parts,
                        False,
                    ),
                ]
                for frame, path, collection, compressed in timing_outputs:
                    if frame is None or frame.empty:
                        continue
                    frame.to_csv(
                        path,
                        index=False,
                        compression="gzip" if compressed else None,
                    )
                    collection.append(path)
                del (
                    timing_summary,
                    timing_profiles,
                    timing_null,
                    timing_bootstrap,
                    timing_lopo,
                    timing_direction,
                    timing_simulation,
                )
                gc.collect()

            real_scores = score_boundary_collection(
                real_obs,
                specs,
                window_sec,
                rr_threshold,
                config.min_segment_points,
            )
            real_scores["MatchedRealBoundaryID"] = real_scores["BoundaryID"]
            pseudo_scores = score_boundary_collection(
                pseudo_obs,
                specs,
                window_sec,
                rr_threshold,
                config.min_segment_points,
            )
            pseudo_expanded = expand_pseudo_results(pseudo_scores, matches)
            logging.info("Cross-fitting score thresholds for %s", combo_label)
            score_evaluation, score_thresholds = crossfit_score_calibration(
                real_scores, pseudo_expanded, config.calibration_fpr
            )
            logging.info("Completed score calibration for %s", combo_label)

            real_pelt_grid = pelt_grid_boundary_collection(
                real_obs,
                specs,
                window_sec,
                rr_threshold,
                config.min_segment_points,
                config.pelt_multipliers,
                config.require_ruptures,
                representations=("reduced",),
            )
            real_pelt_grid["MatchedRealBoundaryID"] = real_pelt_grid["BoundaryID"]
            pseudo_pelt_grid = pelt_grid_boundary_collection(
                pseudo_obs,
                specs,
                window_sec,
                rr_threshold,
                config.min_segment_points,
                config.pelt_multipliers,
                config.require_ruptures,
                representations=("reduced",),
            )
            pseudo_pelt_expanded = expand_pseudo_results(pseudo_pelt_grid, matches)
            logging.info("Cross-fitting PELT penalties for %s", combo_label)
            pelt_evaluation, pelt_calibration = crossfit_pelt_calibration(
                real_pelt_grid,
                pseudo_pelt_expanded,
                config.calibration_fpr,
                config.pelt_multipliers,
            )
            logging.info("Completed PELT calibration for %s", combo_label)
            evaluation = pd.concat(
                [frame for frame in [score_evaluation, pelt_evaluation] if not frame.empty],
                ignore_index=True,
            )
            all_thresholds.append(score_thresholds)
            all_pelt_calibration.append(pelt_calibration)

            logging.info("Computing matched empirical P values for %s", combo_label)
            empirical = matched_empirical_pvalues(real_scores, pseudo_expanded)
            logging.info("Completed matched empirical P values for %s", combo_label)

            evaluation_path = boundary_parts_dir / f"evaluation_{combo_label}.csv.gz"
            raw_real_path = boundary_parts_dir / f"raw_real_scores_{combo_label}.csv.gz"
            raw_pseudo_path = boundary_parts_dir / f"raw_pseudo_candidate_scores_{combo_label}.csv.gz"
            empirical_path = inference_parts_dir / f"empirical_pvalues_{combo_label}.csv.gz"
            effects_path = boundary_parts_dir / f"feature_effects_{combo_label}.csv.gz"

            # Write and release the large detector tables before feature effects
            # are calculated. This keeps one configuration within modest memory.
            evaluation.to_csv(evaluation_path, index=False, compression="gzip")
            real_scores.to_csv(raw_real_path, index=False, compression="gzip")
            pseudo_scores.to_csv(raw_pseudo_path, index=False, compression="gzip")
            empirical.to_csv(empirical_path, index=False, compression="gzip")

            evaluation_parts.append(evaluation_path)
            raw_real_parts.append(raw_real_path)
            raw_pseudo_candidate_parts.append(raw_pseudo_path)
            empirical_parts.append(empirical_path)

            del (
                real_scores, pseudo_scores, pseudo_expanded, score_evaluation,
                real_pelt_grid, pseudo_pelt_grid, pseudo_pelt_expanded,
                pelt_evaluation, evaluation, empirical,
            )
            gc.collect()

            logging.info("Computing physiological effect magnitudes for %s", combo_label)
            real_effects = compute_feature_effects(real_obs, window_sec, rr_threshold)
            pseudo_effects = compute_feature_effects(pseudo_obs, window_sec, rr_threshold)
            pseudo_effects_compact = _expand_pseudo_effects(pseudo_effects, matches)
            effects = pd.concat(
                [frame for frame in [real_effects, pseudo_effects_compact] if not frame.empty],
                ignore_index=True,
            )
            effects.to_csv(effects_path, index=False, compression="gzip")
            effect_parts.append(effects_path)

            # Explicitly release the current combination before the next window
            # or quality threshold is prepared.
            del (
                real_obs, pseudo_obs, matches, candidates, real_effects,
                pseudo_effects, pseudo_effects_compact, effects,
            )
            gc.collect()



    thresholds_all = pd.concat(all_thresholds, ignore_index=True)
    pelt_calibration_all = pd.concat(all_pelt_calibration, ignore_index=True)
    real_eligibility_all = pd.concat(all_real_eligibility, ignore_index=True)
    pseudo_audit_all = pd.concat(all_pseudo_audits, ignore_index=True)
    match_audit_all = pd.concat(all_match_audits, ignore_index=True)
    normalization_audit_all = pd.concat(all_normalization_audits, ignore_index=True)
    rep_audit_all = pd.concat(all_rep_audits, ignore_index=True)
    sim_thresholds_all = pd.concat(all_sim_thresholds, ignore_index=True) if all_sim_thresholds else pd.DataFrame()
    sim_legacy_all = pd.concat(all_sim_legacy, ignore_index=True) if all_sim_legacy else pd.DataFrame()
    sim_pelt_all = pd.concat(all_sim_pelt, ignore_index=True) if all_sim_pelt else pd.DataFrame()
    sim_power_all = pd.concat(all_sim_power, ignore_index=True) if all_sim_power else pd.DataFrame()

    # Create convenient combined files by streaming the per-configuration parts;
    # no multi-million-row table is materialized in memory.
    combine_gzip_csv_parts(
        evaluation_parts,
        boundary_dir / "evaluation_results_crossfitted_v4.csv.gz",
    )
    combine_gzip_csv_parts(
        raw_real_parts,
        boundary_dir / "raw_real_scores_v4.csv.gz",
    )
    combine_gzip_csv_parts(
        raw_pseudo_candidate_parts,
        boundary_dir / "raw_pseudo_candidate_scores_v4.csv.gz",
    )
    combine_gzip_csv_parts(
        empirical_parts,
        inference_dir / "boundary_level_matched_empirical_pvalues.csv.gz",
    )
    combine_gzip_csv_parts(
        effect_parts,
        boundary_dir / "feature_effects_real_and_matched_pseudo.csv.gz",
    )
    combine_plain_csv_parts(
        timing_summary_parts,
        inference_dir / "population_shared_timing_summary.csv",
    )
    combine_plain_csv_parts(
        timing_profile_parts,
        inference_dir / "population_shared_timing_profiles.csv",
    )
    combine_gzip_csv_parts(
        timing_null_parts,
        inference_dir / "population_shared_timing_pseudo_null.csv.gz",
    )
    combine_gzip_csv_parts(
        timing_bootstrap_parts,
        inference_dir / "population_shared_timing_bootstrap.csv.gz",
    )
    combine_plain_csv_parts(
        timing_lopo_parts,
        sensitivity_dir / "population_shared_timing_lopo.csv",
    )
    combine_plain_csv_parts(
        timing_direction_parts,
        inference_dir / "population_shared_timing_direction_comparison.csv",
    )
    combine_plain_csv_parts(
        timing_simulation_parts,
        calibration_dir / "population_shared_timing_simulation_power.csv",
    )

    thresholds_all.to_csv(calibration_dir / "direction_specific_crossfit_thresholds.csv", index=False)
    pelt_calibration_all.to_csv(calibration_dir / "direction_specific_pelt_crossfit.csv", index=False)
    real_eligibility_all.to_csv(sensitivity_dir / "real_boundary_eligibility.csv", index=False)
    pseudo_audit_all.to_csv(audit_dir / "pseudo_candidate_eligibility_audit.csv", index=False)
    match_audit_all.to_csv(audit_dir / "pseudo_matching_audit.csv", index=False)
    normalization_audit_all.to_csv(audit_dir / "symmetric_crossfit_normalization_audit.csv", index=False)
    rep_audit_all.to_csv(representation_dir / "representation_statistical_audit_all.csv", index=False)
    if not sim_thresholds_all.empty:
        sim_thresholds_all.to_csv(calibration_dir / "simulation_thresholds_all.csv", index=False)
        sim_legacy_all.to_csv(calibration_dir / "legacy_fixed6_simulated_null_fpr.csv", index=False)
        sim_pelt_all.to_csv(calibration_dir / "pelt_simulation_penalty_grid_all.csv", index=False)
        sim_power_all.to_csv(calibration_dir / "simulation_power_timing_all.csv", index=False)

    common_required = len(config.windows) * len(config.rr_thresholds)
    common_counts = real_eligibility_all.groupby("BoundaryID")["Eligible"].agg(
        EligibleSum="sum", ConfigurationCount="count"
    )
    common_ids = common_counts[
        common_counts["EligibleSum"].eq(common_required)
        & common_counts["ConfigurationCount"].eq(common_required)
    ].index
    inventory[inventory["BoundaryID"].isin(common_ids)].to_csv(
        sensitivity_dir / "common_real_boundaries_all_windows_quality.csv", index=False
    )

    if config.run_lopo:
        logging.info("Starting genuine end-to-end leave-one-participant-out validation")
        run_end_to_end_lopo(data, inventory, config, lopo_dir)

    # The lightweight top-level launcher starts post-processing only after this
    # construction process exits. This avoids inheriting large NumPy/pandas/BLAS
    # state into the participant-bootstrap and reporting process.
    write_json(
        audit_dir / "postprocess_request.json",
        {
            "output_root": str(config.output_root),
            "bootstrap": config.bootstrap_replicates,
            "permutations": config.permutation_replicates,
            "seed": config.seed,
            "figure_dpi": config.figure_dpi,
        },
    )
    logging.info("Completed V4 construction stage; post-processing request saved")
    # Force the construction stage to terminate before Python decreferences the
    # very large collection of boundary, calibration, and simulation tables.
    # Every deliverable needed by post-processing has already been written.
    os._exit(0)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()
