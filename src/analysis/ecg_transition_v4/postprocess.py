from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .inference import (
    compare_directions_participant_level,
    compare_real_pseudo,
    run_direction_models,
    run_feature_direction_models,
    summarize_boundary_order,
    summarize_detection_with_participant_ci,
    summarize_feature_effects,
)
from .reporting import make_figures, write_paper_tables, write_readme
from .utils import ensure_dir


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required post-processing input is missing: {path}")
    return pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])


def _primary_summary_text(
    detection_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    empirical: pd.DataFrame,
    population_timing: pd.DataFrame,
    config_payload: dict[str, object] | None = None,
) -> str:
    config_payload = config_payload or {}
    timing_primary_window = int(config_payload.get("timing_primary_window", 30))
    timing_primary_rr = float(
        config_payload.get("timing_primary_rr_threshold", 20.0)
    )
    timing_primary_representation = str(
        config_payload.get("timing_primary_representation", "reduced")
    )
    timing_primary_endpoint = str(
        config_payload.get("timing_primary_endpoint", "departure_magnitude")
    )

    lines = [
        "PRIMARY NATURE-ORIENTED VALIDATED RESULTS V4",
        "=============================================",
        "",
        "Primary individual-boundary method: direction-specific, pseudo-block-cross-fitted covariance-aware information criterion",
        "Primary population timing method: participant-pooled shared segmented departure trajectory calibrated against stable A and stable NA pseudo controls",
        "Primary representation: reduced nonredundant direct-measure feature set",
        "Primary timing search: post-boundary 0 to +60 seconds",
        "Primary quality threshold: RR correction <=20%",
        "Interpretation: video-session-boundary-associated cardiovascular change; not anxiety-specific.",
        "",
    ]
    primary = detection_summary[
        detection_summary["RRThreshold"].eq(20.0)
        & detection_summary["Representation"].eq("reduced")
        & detection_summary["SearchWindow"].eq("post_only")
        & detection_summary["Method"].eq("CovIC_crossfit")
    ]
    for window in sorted(primary["WindowLength_sec"].dropna().unique()):
        lines.append(f"Window {int(window)} s")
        group = primary[primary["WindowLength_sec"].eq(window)]
        for direction in ["NA_to_A", "A_to_NA"]:
            real = group[
                group["BoundaryKind"].eq("Real")
                & group["TransitionType"].eq(direction)
            ]
            if not real.empty:
                row = real.iloc[0]
                lines.append(
                    f"  Real {direction}: {100*row.DetectionRate:.1f}% "
                    f"(participant-bootstrap 95% CI {100*row.DetectionCI95_Lower:.1f}-"
                    f"{100*row.DetectionCI95_Upper:.1f}%), "
                    f"median detected window centre={row.MedianLatencyDetected_sec} s"
                )
            for condition in ["A", "NA"]:
                pseudo = group[
                    group["BoundaryKind"].eq("Pseudo")
                    & group["TransitionType"].eq(direction)
                    & group["PseudoCondition"].eq(condition)
                ]
                if not pseudo.empty:
                    row = pseudo.iloc[0]
                    lines.append(
                        f"  Cross-fitted pseudo {condition}, {direction}: "
                        f"{100*row.DetectionRate:.1f}%"
                    )
        lines.append("")
    if not comparisons.empty:
        lines.append("Matched real-versus-pseudo risk differences")
        subset = comparisons[
            comparisons["RRThreshold"].eq(20.0)
            & comparisons["Representation"].eq("reduced")
            & comparisons["SearchWindow"].eq("post_only")
            & comparisons["Method"].eq("CovIC_crossfit")
        ]
        for row in subset.sort_values(
            ["WindowLength_sec", "TransitionType", "PseudoComparison"]
        ).itertuples(index=False):
            lines.append(
                f"  {row.WindowLength_sec}s, {row.TransitionType}, pseudo {row.PseudoComparison}: "
                f"RD={row.RiskDifference:.3f} "
                f"({row.RiskDifferenceCI_Lower:.3f} to {row.RiskDifferenceCI_Upper:.3f})"
            )
    if not empirical.empty:
        subset = empirical[
            empirical["RRThreshold"].eq(20.0)
            & empirical["Representation"].eq("reduced")
            & empirical["SearchWindow"].eq("post_only")
            & empirical["Method"].eq("CovIC")
        ]
        lines.extend(
            [
                "",
                f"Boundary-level matched empirical tests with BH q<0.05: "
                f"{int(subset['EmpiricalQValue_BH'].lt(0.05).sum())} of "
                f"{subset['BoundaryID'].nunique()} real boundaries.",
            ]
        )
    if population_timing is not None and not population_timing.empty:
        timing = population_timing[
            population_timing["RRThreshold"].eq(timing_primary_rr)
            & population_timing["Representation"].eq(timing_primary_representation)
            & population_timing["Endpoint"].eq(timing_primary_endpoint)
            & population_timing["SearchWindow"].eq("post_only")
        ].copy()
        if not timing.empty:
            lines.extend(
                [
                    "",
                    "Participant-pooled candidate timing",
                ]
            )
            for row in timing.sort_values(
                ["WindowLength_sec", "TransitionType"]
            ).itertuples(index=False):
                lines.append(
                    f"  {int(row.WindowLength_sec)}s"
                    f"{' [primary]' if int(row.WindowLength_sec) == timing_primary_window else ''}, "
                    f"{row.TransitionType}: "
                    f"candidate centre={row.CandidateTime_sec:g} s "
                    f"(participant-bootstrap 95% interval "
                    f"{row.CandidateTimeCI95_Lower_sec:g}-"
                    f"{row.CandidateTimeCI95_Upper_sec:g} s); "
                    f"pseudo-A p={row.EmpiricalPValue_PseudoA:.4g}, "
                    f"pseudo-NA p={row.EmpiricalPValue_PseudoNA:.4g}; "
                    f"status={row.TimingStatus}."
                )
            lines.append(
                "Candidate timings are always reported when estimable. They are "
                "called validated timings only when the group score exceeds both "
                "stable-session pseudo-control null distributions."
            )
    return "\n".join(lines) + "\n"


def run_postprocess(
    output_root: Path,
    bootstrap: int,
    permutations: int,
    seed: int,
    figure_dpi: int,
) -> None:
    output_root = output_root.expanduser().resolve()
    inference_dir = ensure_dir(output_root / "04_inference")
    sensitivity_dir = ensure_dir(output_root / "05_sensitivity")
    lopo_dir = ensure_dir(output_root / "06_end_to_end_lopo")
    figures_dir = ensure_dir(output_root / "07_figures")
    paper_dir = ensure_dir(output_root / "08_paper_tables")
    calibration_dir = output_root / "02_calibration"
    boundary_dir = output_root / "03_boundary_results"
    audit_dir = output_root / "00_audit"
    representation_dir = output_root / "01_feature_representations"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_root / "timing_v4_postprocess.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    for noisy in ["matplotlib", "fontTools", "PIL", "openpyxl", "numba"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.info("Starting clean-process V4 post-processing")

    evaluation_parts = sorted((boundary_dir / "_parts").glob("evaluation_*.csv.gz"))
    if not evaluation_parts:
        evaluation_parts = [boundary_dir / "evaluation_results_crossfitted_v4.csv.gz"]
    effect_parts = sorted((boundary_dir / "_parts").glob("feature_effects_*.csv.gz"))
    if not effect_parts:
        effect_parts = [boundary_dir / "feature_effects_real_and_matched_pseudo.csv.gz"]
    empirical_parts = sorted((inference_dir / "_parts").glob("empirical_pvalues_*.csv.gz"))
    if not empirical_parts:
        empirical_parts = [inference_dir / "boundary_level_matched_empirical_pvalues.csv.gz"]

    eligibility = _read_required(sensitivity_dir / "real_boundary_eligibility.csv")

    detection_tables: list[pd.DataFrame] = []
    comparison_tables: list[pd.DataFrame] = []
    permutation_tables: list[pd.DataFrame] = []
    boundary_order_tables: list[pd.DataFrame] = []
    direction_model_tables: list[pd.DataFrame] = []
    direction_paired_tables: list[pd.DataFrame] = []

    evaluation_columns = [
        "BoundaryID", "MatchedRealBoundaryID", "BoundaryKind", "Subject",
        "TransitionType", "BoundaryOrder", "PseudoCondition", "CandidateBlockID",
        "MatchScore", "WindowLength_sec", "RRThreshold", "Representation",
        "SearchWindow", "Method", "Detected", "Score", "Latency_sec",
        "MahalanobisMagnitude", "MeanAbsStandardizedChange",
        "SignedFirstDimensionChange",
    ]
    for part_index, part in enumerate(evaluation_parts):
        logging.info("Post-processing detector part %s/%s: %s", part_index + 1, len(evaluation_parts), part.name)
        header = pd.read_csv(part, nrows=0).columns
        usecols = [column for column in evaluation_columns if column in header]
        evaluation = pd.read_csv(
            part, usecols=usecols, low_memory=False,
            keep_default_na=False, na_values=[""],
        )
        detection_tables.append(
            summarize_detection_with_participant_ci(
                evaluation, bootstrap, seed + 1000 + part_index * 10000
            )
        )
        comparisons, permutations_table = compare_real_pseudo(
            evaluation,
            bootstrap,
            permutations,
            seed + 2000 + part_index * 10000,
        )
        comparison_tables.append(comparisons)
        permutation_tables.append(permutations_table)
        boundary_order_tables.append(
            summarize_boundary_order(
                evaluation,
                "CovIC_crossfit",
                "reduced",
                "post_only",
                bootstrap,
                seed + 4000 + part_index * 10000,
            )
        )
        direction_model_tables.append(
            run_direction_models(
                evaluation, "CovIC_crossfit", "reduced", "post_only"
            )
        )
        direction_paired_tables.append(
            compare_directions_participant_level(
                evaluation,
                "CovIC_crossfit",
                "reduced",
                "post_only",
                bootstrap,
                permutations,
                seed + 4500 + part_index * 10000,
            )
        )
        del evaluation

    detection_summary = pd.concat(
        [frame for frame in detection_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in detection_tables) else pd.DataFrame()
    comparisons = pd.concat(
        [frame for frame in comparison_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in comparison_tables) else pd.DataFrame()
    permutation_results = pd.concat(
        [frame for frame in permutation_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in permutation_tables) else pd.DataFrame()
    boundary_order = pd.concat(
        [frame for frame in boundary_order_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in boundary_order_tables) else pd.DataFrame()
    direction_models = pd.concat(
        [frame for frame in direction_model_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in direction_model_tables) else pd.DataFrame()
    direction_paired = pd.concat(
        [frame for frame in direction_paired_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in direction_paired_tables) else pd.DataFrame()

    feature_summary_tables: list[pd.DataFrame] = []
    feature_model_tables: list[pd.DataFrame] = []
    for part_index, part in enumerate(effect_parts):
        logging.info("Post-processing feature-effect part %s/%s: %s", part_index + 1, len(effect_parts), part.name)
        effects = pd.read_csv(
            part, low_memory=False, keep_default_na=False, na_values=[""]
        )
        feature_summary_tables.append(
            summarize_feature_effects(
                effects, bootstrap, seed + 3000 + part_index * 10000
            )
        )
        feature_model_tables.append(run_feature_direction_models(effects))
        del effects
    feature_summary = pd.concat(
        [frame for frame in feature_summary_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in feature_summary_tables) else pd.DataFrame()
    feature_direction_models = pd.concat(
        [frame for frame in feature_model_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in feature_model_tables) else pd.DataFrame()

    empirical_tables: list[pd.DataFrame] = []
    empirical_primary_tables: list[pd.DataFrame] = []
    for part in empirical_parts:
        empirical = _read_required(part)
        empirical_tables.append(empirical)
        empirical_primary_tables.append(
            empirical[
                empirical["RRThreshold"].eq(20.0)
                & empirical["Representation"].eq("reduced")
                & empirical["SearchWindow"].eq("post_only")
                & empirical["Method"].eq("CovIC")
            ].copy()
        )
    empirical_all = pd.concat(empirical_tables, ignore_index=True)
    empirical_primary = pd.concat(
        [frame for frame in empirical_primary_tables if not frame.empty], ignore_index=True
    ) if any(not frame.empty for frame in empirical_primary_tables) else pd.DataFrame()

    detection_summary.to_csv(
        inference_dir / "detection_rates_participant_bootstrap.csv", index=False
    )
    comparisons.to_csv(
        inference_dir / "real_vs_pseudo_effect_sizes.csv", index=False
    )
    permutation_results.to_csv(
        inference_dir / "participant_sign_flip_tests.csv", index=False
    )
    feature_summary.to_csv(
        inference_dir / "feature_magnitude_direction_summary.csv", index=False
    )
    boundary_order.to_csv(
        sensitivity_dir / "boundary_order_sequence_position.csv", index=False
    )
    direction_models.to_csv(
        inference_dir / "direction_timing_magnitude_GEE.csv", index=False
    )
    direction_paired.to_csv(
        inference_dir / "direction_comparison_participant_bootstrap.csv", index=False
    )
    feature_direction_models.to_csv(
        inference_dir / "feature_direction_GEE.csv", index=False
    )

    empirical_summary = (
        empirical_all.groupby(
            [
                "WindowLength_sec",
                "RRThreshold",
                "Representation",
                "SearchWindow",
                "Method",
                "TransitionType",
                "PseudoCondition",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            RealBoundaries=("BoundaryID", "nunique"),
            MedianPseudoControls=("PseudoControls", "median"),
            MedianEmpiricalPValue=("EmpiricalPValue", "median"),
            MinimumEmpiricalPValue=("EmpiricalPValue", "min"),
            BoundariesPBelow05=("EmpiricalPValue", lambda x: int((x < 0.05).sum())),
            BoundariesQBelow05=("EmpiricalQValue_BH", lambda x: int((x < 0.05).sum())),
        )
    )
    empirical_summary.to_csv(
        inference_dir / "boundary_level_empirical_pvalue_summary.csv", index=False
    )

    thresholds = _read_optional(
        calibration_dir / "direction_specific_crossfit_thresholds.csv"
    )
    pelt_calibration = _read_optional(
        calibration_dir / "direction_specific_pelt_crossfit.csv"
    )
    sim_thresholds = _read_optional(
        calibration_dir / "simulation_thresholds_all.csv"
    )
    sim_power = _read_optional(
        calibration_dir / "simulation_power_timing_all.csv"
    )
    quality_window = _read_optional(
        sensitivity_dir / "rr_quality_window_summary.csv"
    )
    match_audit = _read_optional(audit_dir / "pseudo_matching_audit.csv")
    rep_audit = _read_optional(
        representation_dir / "representation_statistical_audit_all.csv"
    )
    lopo_summary = _read_optional(lopo_dir / "end_to_end_lopo_summary.csv")
    lopo_audit = _read_optional(lopo_dir / "end_to_end_lopo_fold_audit.csv")
    population_timing_summary = _read_optional(
        inference_dir / "population_shared_timing_summary.csv"
    )
    population_timing_profiles = _read_optional(
        inference_dir / "population_shared_timing_profiles.csv"
    )
    population_timing_direction = _read_optional(
        inference_dir / "population_shared_timing_direction_comparison.csv"
    )
    population_timing_lopo = _read_optional(
        sensitivity_dir / "population_shared_timing_lopo.csv"
    )
    population_timing_simulation = _read_optional(
        calibration_dir / "population_shared_timing_simulation_power.csv"
    )

    manifest_path = audit_dir / "analysis_manifest.json"
    config_payload: dict[str, object] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_payload = dict(payload.get("config", {}))

    paper_tables = {
        "Table1_Real_vs_Pseudo": comparisons,
        "Table2_Detection_Rates": detection_summary,
        "Table3_Empirical_PValues": empirical_summary,
        "Table4_Crossfit_Thresholds": thresholds,
        "Table5_PELT_Calibration": pelt_calibration,
        "Table6_Simulation": sim_thresholds,
        "Table7_RR_Quality": quality_window,
        "Table8_Boundary_Order": boundary_order,
        "Table9_Direction_Paired": direction_paired,
        "Table10_Direction_GEE": direction_models,
        "Table11_Feature_Changes": feature_summary,
        "Table12_LOPO": lopo_summary,
        "Table13_LOPO_Audit": lopo_audit,
        "Table14_Pseudo_Matching": match_audit,
        "Table15_Representation_Audit": rep_audit,
        "Table16_Population_Timing": population_timing_summary,
        "Table17_Timing_Direction": population_timing_direction,
        "Table18_Timing_LOPO": population_timing_lopo,
        "Table19_Timing_Simulation": population_timing_simulation,
    }
    workbook = write_paper_tables(paper_dir, paper_tables)
    make_figures(
        figures_dir,
        pd.DataFrame(),
        detection_summary,
        comparisons,
        sim_power,
        boundary_order,
        eligibility,
        feature_summary,
        empirical_primary,
        figure_dpi,
        population_timing_summary=population_timing_summary,
        population_timing_profiles=population_timing_profiles,
        population_timing_simulation=population_timing_simulation,
    )

    summary = _primary_summary_text(
        detection_summary,
        comparisons,
        empirical_primary,
        population_timing_summary,
        config_payload,
    )
    (output_root / "PRIMARY_RESULTS_SUMMARY_V4.txt").write_text(
        summary, encoding="utf-8"
    )

    config_object = SimpleNamespace(**config_payload)
    write_readme(
        output_root,
        "ecg_transition_analysis_timing_v4.py",
        config_object,
        workbook,
    )
    logging.info("Completed V4 timing post-processing")
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process Nature-oriented V4 outputs")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, required=True)
    parser.add_argument("--permutations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--figure-dpi", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_postprocess(
        args.output_root,
        args.bootstrap,
        args.permutations,
        args.seed,
        args.figure_dpi,
    )


if __name__ == "__main__":
    main()
