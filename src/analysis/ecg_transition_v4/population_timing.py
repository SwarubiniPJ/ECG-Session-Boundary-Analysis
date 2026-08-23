from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import RepresentationSpec, RunConfig, SEARCH_WINDOWS
from .utils import quantile_higher


@dataclass(frozen=True)
class TimingFit:
    score: float
    raw_gain: float
    candidate_time_sec: float
    breakpoint_midpoint_sec: float
    interval_left_sec: float
    interval_right_sec: float
    window_support_start_sec: float
    window_support_end_sec: float
    sse_no_change: float
    sse_segmented: float
    level_change_norm: float
    slope_change_norm: float
    signed_first_level_change: float
    signed_first_slope_change: float
    participants: int
    participant_time_rows: int
    candidate_count: int


class SharedTimingFitter:
    """Fit one participant-balanced shared segmented trajectory.

    The participant-time trajectory is fitted with participant fixed intercepts,
    a common linear time trend, and candidate shared level/slope changes. The
    score is the participant-normalized reduction in weighted multivariate SSE.
    It is not assigned a parametric P value. Statistical support is determined
    entirely from matched stable-session pseudo-boundary group draws.
    """

    def __init__(
        self,
        window_sec: int,
        min_unique_times_per_side: int,
        min_participants: int,
    ) -> None:
        self.window_sec = int(window_sec)
        self.min_unique_times_per_side = int(min_unique_times_per_side)
        self.min_participants = int(min_participants)

    @staticmethod
    def _participant_equal_weights(frame: pd.DataFrame) -> NDArray[np.float64]:
        counts = frame.groupby("Subject", sort=False).size().to_dict()
        return np.asarray(
            [1.0 / max(int(counts[str(subject)]), 1) for subject in frame["Subject"]],
            dtype=float,
        )

    @staticmethod
    def _weighted_sse(
        values: NDArray[np.float64],
        design: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        square_root = np.sqrt(weights)[:, None]
        weighted_design = design * square_root
        weighted_values = values * square_root
        beta, _, _, _ = np.linalg.lstsq(
            weighted_design,
            weighted_values,
            rcond=None,
        )
        residuals = values - design @ beta
        sse = float(np.sum(weights[:, None] * residuals * residuals))
        return sse, beta

    def fit(
        self,
        trajectory: pd.DataFrame,
        value_columns: Sequence[str],
        search_start_sec: float,
        search_end_sec: float,
    ) -> tuple[TimingFit | None, pd.DataFrame]:
        required = ["Subject", "RelativeCenter_sec", *value_columns]
        if trajectory.empty or any(column not in trajectory.columns for column in required):
            return None, pd.DataFrame()
        data = trajectory[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        data["Subject"] = data["Subject"].astype(str)
        participants = sorted(data["Subject"].unique())
        if len(participants) < self.min_participants:
            return None, pd.DataFrame()

        times_sec = data["RelativeCenter_sec"].to_numpy(dtype=float)
        times_min = times_sec / 60.0
        values = data[list(value_columns)].to_numpy(dtype=float)
        weights = self._participant_equal_weights(data)
        participant_dummies = pd.get_dummies(
            data["Subject"],
            drop_first=False,
            dtype=float,
        ).to_numpy(dtype=float)

        base_design = np.column_stack([participant_dummies, times_min])
        sse_no_change, _ = self._weighted_sse(values, base_design, weights)

        unique_times = np.sort(np.unique(times_sec))
        candidates: list[dict[str, float | int]] = []
        for candidate_time in unique_times:
            if candidate_time < float(search_start_sec) - 1e-9:
                continue
            if candidate_time > float(search_end_sec) + 1e-9:
                continue
            left_unique = int(np.sum(unique_times < candidate_time))
            right_unique = int(np.sum(unique_times >= candidate_time))
            if left_unique < self.min_unique_times_per_side:
                continue
            if right_unique < self.min_unique_times_per_side:
                continue

            tau_min = float(candidate_time) / 60.0
            step = (times_min >= tau_min).astype(float)
            hinge = np.maximum(times_min - tau_min, 0.0)
            segmented_design = np.column_stack(
                [participant_dummies, times_min, step, hinge]
            )
            sse_segmented, beta = self._weighted_sse(
                values,
                segmented_design,
                weights,
            )
            raw_gain = float(sse_no_change - sse_segmented)
            normalized_score = raw_gain / max(len(participants), 1)
            level = np.asarray(beta[-2], dtype=float).reshape(-1)
            slope = np.asarray(beta[-1], dtype=float).reshape(-1)
            candidates.append(
                {
                    "CandidateTime_sec": float(candidate_time),
                    "Score": normalized_score,
                    "RawGain": raw_gain,
                    "SSE_NoChange": sse_no_change,
                    "SSE_Segmented": sse_segmented,
                    "LevelChangeNorm": float(np.linalg.norm(level)),
                    "SlopeChangeNorm": float(np.linalg.norm(slope)),
                    "SignedFirstLevelChange": (
                        float(level[0]) if level.size else np.nan
                    ),
                    "SignedFirstSlopeChange": (
                        float(slope[0]) if slope.size else np.nan
                    ),
                    "Participants": len(participants),
                    "ParticipantTimeRows": len(data),
                    "UniqueTimesLeft": left_unique,
                    "UniqueTimesRight": right_unique,
                }
            )

        profile = pd.DataFrame(candidates)
        if profile.empty:
            return None, profile
        best_index = int(profile["Score"].idxmax())
        best = profile.loc[best_index]
        candidate_time = float(best["CandidateTime_sec"])
        earlier = unique_times[unique_times < candidate_time]
        last_old = float(np.max(earlier)) if earlier.size else candidate_time
        midpoint = float((last_old + candidate_time) / 2.0)
        fit = TimingFit(
            score=float(best["Score"]),
            raw_gain=float(best["RawGain"]),
            candidate_time_sec=candidate_time,
            breakpoint_midpoint_sec=midpoint,
            interval_left_sec=last_old,
            interval_right_sec=candidate_time,
            window_support_start_sec=candidate_time - self.window_sec / 2.0,
            window_support_end_sec=candidate_time + self.window_sec / 2.0,
            sse_no_change=float(best["SSE_NoChange"]),
            sse_segmented=float(best["SSE_Segmented"]),
            level_change_norm=float(best["LevelChangeNorm"]),
            slope_change_norm=float(best["SlopeChangeNorm"]),
            signed_first_level_change=float(best["SignedFirstLevelChange"]),
            signed_first_slope_change=float(best["SignedFirstSlopeChange"]),
            participants=len(participants),
            participant_time_rows=len(data),
            candidate_count=len(profile),
        )
        return fit, profile


def _endpoint_columns(endpoint: str, dimensions: int) -> list[str]:
    if endpoint == "departure_magnitude":
        return ["TimingValue"]
    if endpoint == "signed_trajectory":
        return [f"TimingValue{index + 1}" for index in range(dimensions)]
    raise ValueError(f"Unsupported population timing endpoint: {endpoint}")


def prepare_endpoint_observations(
    observations: pd.DataFrame,
    spec: RepresentationSpec,
    endpoint: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Transform boundary observations into a timing endpoint.

    ``signed_trajectory`` retains the covariance-whitened representation.
    ``departure_magnitude`` converts each time point to its Euclidean distance
    from that boundary's mean pre-boundary state in whitened coordinates. The
    latter preserves changes that differ in direction across participants.
    """
    if observations.empty:
        return pd.DataFrame(), _endpoint_columns(endpoint, len(spec.output_columns))
    metadata_columns = [
        column
        for column in [
            "BoundaryID",
            "Subject",
            "TransitionType",
            "BoundaryOrder",
            "CandidateID",
            "CandidateBlockID",
            "PseudoCondition",
            "PseudoFold",
            "RelativeCenter_sec",
        ]
        if column in observations.columns
    ]
    output = observations[metadata_columns].reset_index(drop=True).copy()
    transformed = spec.transform(observations)
    if not np.isfinite(transformed).all():
        raise ValueError(
            f"Non-finite values entered population timing for {spec.name}."
        )
    whitened = np.asarray(transformed, dtype=float) @ spec.whitening

    if endpoint == "signed_trajectory":
        value_columns = _endpoint_columns(endpoint, whitened.shape[1])
        for index, column in enumerate(value_columns):
            output[column] = whitened[:, index]
        return output, value_columns

    if endpoint != "departure_magnitude":
        raise ValueError(f"Unsupported population timing endpoint: {endpoint}")

    distances = np.full(len(output), np.nan, dtype=float)
    for _, row_indices in output.groupby("BoundaryID", sort=False).groups.items():
        indices = np.asarray(list(row_indices), dtype=int)
        times = output.loc[indices, "RelativeCenter_sec"].to_numpy(dtype=float)
        matrix = whitened[indices]
        pre = matrix[times < 0]
        baseline = (
            np.mean(pre, axis=0)
            if len(pre)
            else np.mean(matrix, axis=0)
        )
        distances[indices] = np.sqrt(
            np.sum((matrix - baseline[None, :]) ** 2, axis=1)
        )
    output["TimingValue"] = distances
    return output, ["TimingValue"]


def participant_average_trajectory(
    endpoint_observations: pd.DataFrame,
    value_columns: Sequence[str],
    direction: str | None = None,
) -> pd.DataFrame:
    data = endpoint_observations.copy()
    if direction is not None and "TransitionType" in data.columns:
        data = data[data["TransitionType"].astype(str).eq(str(direction))]
    if data.empty:
        return pd.DataFrame()
    # First collapse any duplicate rows within a boundary and time, then average
    # repeated boundaries within participant. This gives every boundary equal
    # weight before the participant-balanced segmented fit.
    boundary_time = (
        data.groupby(
            ["BoundaryID", "Subject", "RelativeCenter_sec"],
            as_index=False,
            dropna=False,
        )[list(value_columns)]
        .mean()
    )
    return (
        boundary_time.groupby(
            ["Subject", "RelativeCenter_sec"],
            as_index=False,
            dropna=False,
        )[list(value_columns)]
        .mean()
    )


def _candidate_endpoint_lookup(
    endpoint_observations: pd.DataFrame,
    value_columns: Sequence[str],
) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for candidate_id, group in endpoint_observations.groupby("CandidateID", sort=False):
        lookup[str(candidate_id)] = group[
            ["RelativeCenter_sec", *value_columns]
        ].copy()
    return lookup


def _match_options(
    matches: pd.DataFrame,
    direction: str,
    pseudo_condition: str,
) -> dict[str, dict[str, list[tuple[str, str, float]]]]:
    subset = matches[
        matches["TransitionType"].astype(str).eq(direction)
        & matches["PseudoCondition"].astype(str).eq(pseudo_condition)
    ].copy()
    options: dict[str, dict[str, list[tuple[str, str, float]]]] = {}
    for subject, subject_group in subset.groupby("Subject", sort=True):
        boundary_options: dict[str, list[tuple[str, str, float]]] = {}
        for boundary_id, boundary_group in subject_group.groupby(
            "MatchedRealBoundaryID", sort=True
        ):
            ordered = boundary_group.sort_values(
                ["MatchScore", "MatchRank", "CandidateID"],
                kind="stable",
            )
            boundary_options[str(boundary_id)] = [
                (
                    str(row.CandidateID),
                    str(row.CandidateBlockID),
                    float(row.MatchScore),
                )
                for row in ordered.itertuples(index=False)
            ]
        if boundary_options:
            options[str(subject)] = boundary_options
    return options


def draw_matched_pseudo_trajectory(
    match_options: dict[str, dict[str, list[tuple[str, str, float]]]],
    candidate_lookup: dict[str, pd.DataFrame],
    value_columns: Sequence[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    participant_parts: list[pd.DataFrame] = []
    selected_matches = 0
    skipped_for_scarcity = 0
    participants_with_data = 0

    for subject, boundaries in match_options.items():
        used_blocks: set[str] = set()
        selected_candidates: list[str] = []
        boundary_ids = list(boundaries)
        rng.shuffle(boundary_ids)
        for boundary_id in boundary_ids:
            candidates = boundaries[boundary_id]
            order = rng.permutation(len(candidates))
            choice: tuple[str, str, float] | None = None
            for candidate_index in order:
                candidate = candidates[int(candidate_index)]
                if candidate[1] not in used_blocks:
                    choice = candidate
                    break
            if choice is None:
                # Reusing a finite pseudo block as if it were an independent
                # repeated transition would inflate information. Skip this real
                # boundary in the current null draw instead.
                skipped_for_scarcity += 1
                continue
            used_blocks.add(choice[1])
            selected_candidates.append(choice[0])
            selected_matches += 1

        pieces: list[pd.DataFrame] = []
        for candidate_id in selected_candidates:
            candidate = candidate_lookup.get(candidate_id)
            if candidate is None or candidate.empty:
                continue
            part = candidate.copy()
            part["Subject"] = str(subject)
            pieces.append(part)
        if not pieces:
            continue
        participants_with_data += 1
        participant = pd.concat(pieces, ignore_index=True)
        participant = (
            participant.groupby(
                ["Subject", "RelativeCenter_sec"],
                as_index=False,
            )[list(value_columns)]
            .mean()
        )
        participant_parts.append(participant)

    trajectory = (
        pd.concat(participant_parts, ignore_index=True)
        if participant_parts
        else pd.DataFrame()
    )
    audit = {
        "SelectedMatches": int(selected_matches),
        "SkippedMatchesDueToBlockScarcity": int(skipped_for_scarcity),
        "ParticipantsWithPseudoData": int(participants_with_data),
    }
    return trajectory, audit


def _bootstrap_trajectory(
    trajectory: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    participants = np.asarray(
        sorted(trajectory["Subject"].astype(str).unique()),
        dtype=object,
    )
    if participants.size == 0:
        return pd.DataFrame()
    groups = {
        subject: trajectory[trajectory["Subject"].astype(str).eq(subject)].copy()
        for subject in participants
    }
    sampled = rng.choice(participants, size=len(participants), replace=True)
    parts: list[pd.DataFrame] = []
    for copy_index, subject in enumerate(sampled):
        part = groups[str(subject)].copy()
        part["Subject"] = f"{subject}__bootstrap_{copy_index}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def inject_known_shared_change(
    trajectory: pd.DataFrame,
    value_columns: Sequence[str],
    true_time_sec: float,
    effect_size_sd: float,
    affected_fraction: float,
    shape: str,
    endpoint: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject an abrupt or ramped shared change into a pseudo trajectory."""
    out = trajectory.copy()
    if out.empty:
        return out
    values = out[list(value_columns)].to_numpy(dtype=float)
    times = out["RelativeCenter_sec"].to_numpy(dtype=float)
    pre = values[times < 0]
    scale = np.nanstd(pre, axis=0, ddof=1) if len(pre) > 1 else np.nanstd(values, axis=0)
    fallback = np.nanstd(values, axis=0, ddof=1) if len(values) > 1 else np.ones(values.shape[1])
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)

    dimensions = values.shape[1]
    affected_count = max(1, int(np.ceil(dimensions * float(affected_fraction))))
    affected = rng.choice(np.arange(dimensions), size=affected_count, replace=False)
    signs = (
        np.ones(affected_count, dtype=float)
        if endpoint == "departure_magnitude"
        else rng.choice(np.asarray([-1.0, 1.0]), size=affected_count)
    )
    if shape == "abrupt":
        progress = (times >= float(true_time_sec)).astype(float)
    elif shape == "ramp":
        denominator = max(float(np.max(times) - true_time_sec), 1e-6)
        progress = np.clip((times - float(true_time_sec)) / denominator, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown timing simulation shape: {shape}")

    for index, sign in zip(affected, signs):
        values[:, index] += (
            progress * float(effect_size_sd) * float(scale[index]) * float(sign)
        )
    out.loc[:, list(value_columns)] = values
    return out


def _timing_status(
    p_a: float,
    p_na: float,
    alpha: float,
) -> tuple[str, bool, bool, bool]:
    against_a = bool(np.isfinite(p_a) and p_a < alpha)
    against_na = bool(np.isfinite(p_na) and p_na < alpha)
    against_both = bool(against_a and against_na)
    if against_both:
        status = "validated_against_both_pseudo_controls"
    elif against_a:
        status = "supported_against_pseudo_A_only"
    elif against_na:
        status = "supported_against_pseudo_NA_only"
    else:
        status = "descriptive_candidate_not_validated"
    return status, against_a, against_na, against_both


def _mode_and_probability(values: Iterable[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return np.nan, np.nan
    counts = Counter(finite)
    mode, frequency = counts.most_common(1)[0]
    return float(mode), float(frequency / len(finite))


def _fit_to_dict(fit: TimingFit | None) -> dict[str, object]:
    if fit is None:
        return {
            "PeakScore": np.nan,
            "RawGain": np.nan,
            "CandidateTime_sec": np.nan,
            "BreakpointMidpoint_sec": np.nan,
            "BreakpointIntervalLeft_sec": np.nan,
            "BreakpointIntervalRight_sec": np.nan,
            "WindowSupportStart_sec": np.nan,
            "WindowSupportEnd_sec": np.nan,
            "SSE_NoChange": np.nan,
            "SSE_Segmented": np.nan,
            "LevelChangeNorm": np.nan,
            "SlopeChangeNorm": np.nan,
            "SignedFirstLevelChange": np.nan,
            "SignedFirstSlopeChange": np.nan,
            "Participants": 0,
            "ParticipantTimeRows": 0,
            "CandidateCount": 0,
        }
    return {
        "PeakScore": fit.score,
        "RawGain": fit.raw_gain,
        "CandidateTime_sec": fit.candidate_time_sec,
        "BreakpointMidpoint_sec": fit.breakpoint_midpoint_sec,
        "BreakpointIntervalLeft_sec": fit.interval_left_sec,
        "BreakpointIntervalRight_sec": fit.interval_right_sec,
        "WindowSupportStart_sec": fit.window_support_start_sec,
        "WindowSupportEnd_sec": fit.window_support_end_sec,
        "SSE_NoChange": fit.sse_no_change,
        "SSE_Segmented": fit.sse_segmented,
        "LevelChangeNorm": fit.level_change_norm,
        "SlopeChangeNorm": fit.slope_change_norm,
        "SignedFirstLevelChange": fit.signed_first_level_change,
        "SignedFirstSlopeChange": fit.signed_first_slope_change,
        "Participants": fit.participants,
        "ParticipantTimeRows": fit.participant_time_rows,
        "CandidateCount": fit.candidate_count,
    }


def run_population_shared_timing(
    real_observations: pd.DataFrame,
    pseudo_observations: pd.DataFrame,
    matches: pd.DataFrame,
    specs: dict[str, RepresentationSpec],
    window_sec: int,
    rr_threshold: float,
    config: RunConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Estimate pooled timing for both directions and calibrate with pseudo blocks.

    Returns summary, real score profiles, pseudo null draws, participant bootstrap
    draws, timing leave-one-participant-out results, and direction timing contrasts.
    A candidate time is always reported when the model is estimable. Validation
    against stable pseudo-boundaries is reported separately and is never forced.
    """
    if not config.run_population_timing:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty, empty
    if int(window_sec) not in set(config.timing_windows):
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty, empty
    if float(rr_threshold) not in set(float(value) for value in config.timing_rr_thresholds):
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty, empty

    fitter = SharedTimingFitter(
        window_sec=window_sec,
        min_unique_times_per_side=config.timing_min_unique_times_per_side,
        min_participants=config.timing_min_participants,
    )
    rng = np.random.default_rng(
        config.seed + int(window_sec) * 1009 + int(round(rr_threshold * 10))
    )

    summary_rows: list[dict[str, object]] = []
    real_profile_rows: list[pd.DataFrame] = []
    pseudo_null_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    lopo_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    simulation_rows: list[dict[str, object]] = []

    for representation in config.timing_representations:
        spec = specs.get(representation)
        if spec is None:
            continue
        for endpoint in config.timing_endpoints:
            logging.info(
                "Population timing: window=%ss RR<=%s representation=%s endpoint=%s",
                window_sec,
                rr_threshold,
                representation,
                endpoint,
            )
            real_endpoint, value_columns = prepare_endpoint_observations(
                real_observations,
                spec,
                endpoint,
            )
            pseudo_endpoint, pseudo_value_columns = prepare_endpoint_observations(
                pseudo_observations,
                spec,
                endpoint,
            )
            if value_columns != pseudo_value_columns:
                raise RuntimeError("Real and pseudo timing endpoint dimensions differ.")
            candidate_lookup = _candidate_endpoint_lookup(
                pseudo_endpoint,
                value_columns,
            )

            direction_trajectories: dict[str, pd.DataFrame] = {}
            direction_fits: dict[tuple[str, str], TimingFit | None] = {}
            direction_bootstrap_times: dict[tuple[str, str], list[float]] = {}

            for direction in ["NA_to_A", "A_to_NA"]:
                real_trajectory = participant_average_trajectory(
                    real_endpoint,
                    value_columns,
                    direction,
                )
                direction_trajectories[direction] = real_trajectory

                null_by_condition: dict[str, dict[str, list[float]]] = {
                    condition: {
                        search_name: []
                        for search_name in config.timing_search_windows
                    }
                    for condition in ["A", "NA"]
                }
                null_time_by_condition: dict[str, dict[str, list[float]]] = {
                    condition: {
                        search_name: []
                        for search_name in config.timing_search_windows
                    }
                    for condition in ["A", "NA"]
                }
                pointwise_scores: dict[
                    tuple[str, str, float], list[float]
                ] = defaultdict(list)

                options_by_condition: dict[
                    str, dict[str, dict[str, list[tuple[str, str, float]]]]
                ] = {}
                for pseudo_condition in ["A", "NA"]:
                    options = _match_options(
                        matches,
                        direction,
                        pseudo_condition,
                    )
                    options_by_condition[pseudo_condition] = options
                    for draw_index in range(config.timing_pseudo_draws):
                        pseudo_trajectory, draw_audit = draw_matched_pseudo_trajectory(
                            options,
                            candidate_lookup,
                            value_columns,
                            rng,
                        )
                        for search_name in config.timing_search_windows:
                            search_start, search_end = SEARCH_WINDOWS[search_name]
                            pseudo_fit, pseudo_profile = fitter.fit(
                                pseudo_trajectory,
                                value_columns,
                                search_start,
                                search_end,
                            )
                            if pseudo_fit is None:
                                continue
                            null_by_condition[pseudo_condition][search_name].append(
                                pseudo_fit.score
                            )
                            null_time_by_condition[pseudo_condition][search_name].append(
                                pseudo_fit.candidate_time_sec
                            )
                            for profile_row in pseudo_profile.itertuples(index=False):
                                pointwise_scores[
                                    (
                                        pseudo_condition,
                                        search_name,
                                        float(profile_row.CandidateTime_sec),
                                    )
                                ].append(float(profile_row.Score))
                            pseudo_null_rows.append(
                                {
                                    "WindowLength_sec": int(window_sec),
                                    "RRThreshold": float(rr_threshold),
                                    "Representation": representation,
                                    "Endpoint": endpoint,
                                    "TransitionType": direction,
                                    "PseudoCondition": pseudo_condition,
                                    "SearchWindow": search_name,
                                    "Draw": int(draw_index),
                                    **_fit_to_dict(pseudo_fit),
                                    **draw_audit,
                                }
                            )

                for search_name in config.timing_search_windows:
                    search_start, search_end = SEARCH_WINDOWS[search_name]
                    real_fit, real_profile = fitter.fit(
                        real_trajectory,
                        value_columns,
                        search_start,
                        search_end,
                    )
                    direction_fits[(direction, search_name)] = real_fit
                    if real_fit is None:
                        summary_rows.append(
                            {
                                "WindowLength_sec": int(window_sec),
                                "RRThreshold": float(rr_threshold),
                                "Representation": representation,
                                "Endpoint": endpoint,
                                "TransitionType": direction,
                                "SearchWindow": search_name,
                                "TimingStatus": "not_estimable",
                                **_fit_to_dict(None),
                            }
                        )
                        continue

                    real_profile = real_profile.copy()
                    real_profile.insert(0, "WindowLength_sec", int(window_sec))
                    real_profile.insert(1, "RRThreshold", float(rr_threshold))
                    real_profile.insert(2, "Representation", representation)
                    real_profile.insert(3, "Endpoint", endpoint)
                    real_profile.insert(4, "TransitionType", direction)
                    real_profile.insert(5, "SearchWindow", search_name)
                    for pseudo_condition in ["A", "NA"]:
                        medians: list[float] = []
                        q95_values: list[float] = []
                        for candidate_time in real_profile["CandidateTime_sec"]:
                            values = pointwise_scores.get(
                                (
                                    pseudo_condition,
                                    search_name,
                                    float(candidate_time),
                                ),
                                [],
                            )
                            medians.append(
                                float(np.median(values)) if values else np.nan
                            )
                            q95_values.append(
                                quantile_higher(values, 1.0 - config.timing_alpha)
                                if values
                                else np.nan
                            )
                        real_profile[
                            f"Pseudo{pseudo_condition}_PointwiseMedian"
                        ] = medians
                        real_profile[
                            f"Pseudo{pseudo_condition}_PointwiseQ95"
                        ] = q95_values
                    real_profile_rows.append(real_profile)

                    null_a = np.asarray(
                        null_by_condition["A"][search_name],
                        dtype=float,
                    )
                    null_na = np.asarray(
                        null_by_condition["NA"][search_name],
                        dtype=float,
                    )
                    threshold_a = quantile_higher(
                        null_a,
                        1.0 - config.timing_alpha,
                    )
                    threshold_na = quantile_higher(
                        null_na,
                        1.0 - config.timing_alpha,
                    )
                    p_a = (
                        float(
                            (1 + np.sum(null_a >= real_fit.score))
                            / (len(null_a) + 1)
                        )
                        if len(null_a)
                        else np.nan
                    )
                    p_na = (
                        float(
                            (1 + np.sum(null_na >= real_fit.score))
                            / (len(null_na) + 1)
                        )
                        if len(null_na)
                        else np.nan
                    )
                    status, supported_a, supported_na, supported_both = _timing_status(
                        p_a,
                        p_na,
                        config.timing_alpha,
                    )
                    dual_threshold = float(
                        np.nanmax([threshold_a, threshold_na])
                    )

                    if config.timing_power_simulations > 0:
                        candidate_times = sorted(
                            real_profile["CandidateTime_sec"].dropna().astype(float).unique()
                        )
                        if candidate_times:
                            position_times = {
                                "early": candidate_times[0],
                                "middle": candidate_times[len(candidate_times) // 2],
                                "late": candidate_times[-1],
                            }
                            for pseudo_condition in ["A", "NA"]:
                                simulation_options = options_by_condition[pseudo_condition]
                                for effect_size in config.timing_simulation_effect_sizes:
                                    for shape in ["abrupt", "ramp"]:
                                        for position, true_time in position_times.items():
                                            for simulation_index in range(
                                                config.timing_power_simulations
                                            ):
                                                baseline_trajectory, simulation_audit = (
                                                    draw_matched_pseudo_trajectory(
                                                        simulation_options,
                                                        candidate_lookup,
                                                        value_columns,
                                                        rng,
                                                    )
                                                )
                                                injected = inject_known_shared_change(
                                                    baseline_trajectory,
                                                    value_columns,
                                                    true_time,
                                                    effect_size,
                                                    config.timing_simulation_affected_fraction,
                                                    shape,
                                                    endpoint,
                                                    rng,
                                                )
                                                simulated_fit, _ = fitter.fit(
                                                    injected,
                                                    value_columns,
                                                    search_start,
                                                    search_end,
                                                )
                                                simulation_rows.append(
                                                    {
                                                        "WindowLength_sec": int(window_sec),
                                                        "RRThreshold": float(rr_threshold),
                                                        "Representation": representation,
                                                        "Endpoint": endpoint,
                                                        "TransitionType": direction,
                                                        "PseudoBaselineCondition": pseudo_condition,
                                                        "SearchWindow": search_name,
                                                        "EffectSizeSD": float(effect_size),
                                                        "AffectedFraction": float(
                                                            config.timing_simulation_affected_fraction
                                                        ),
                                                        "ChangeShape": shape,
                                                        "ChangePosition": position,
                                                        "TrueTime_sec": float(true_time),
                                                        "Simulation": int(simulation_index),
                                                        "DetectedAgainstDualPseudoThreshold": bool(
                                                            simulated_fit is not None
                                                            and simulated_fit.score >= dual_threshold
                                                        ),
                                                        "EstimatedTime_sec": (
                                                            simulated_fit.candidate_time_sec
                                                            if simulated_fit is not None
                                                            else np.nan
                                                        ),
                                                        "AbsoluteTimingError_sec": (
                                                            abs(
                                                                simulated_fit.candidate_time_sec
                                                                - float(true_time)
                                                            )
                                                            if simulated_fit is not None
                                                            else np.nan
                                                        ),
                                                        "PeakScore": (
                                                            simulated_fit.score
                                                            if simulated_fit is not None
                                                            else np.nan
                                                        ),
                                                        **simulation_audit,
                                                    }
                                                )

                    bootstrap_times: list[float] = []
                    bootstrap_scores: list[float] = []
                    for bootstrap_index in range(
                        config.timing_bootstrap_replicates
                    ):
                        bootstrap_trajectory = _bootstrap_trajectory(
                            real_trajectory,
                            rng,
                        )
                        bootstrap_fit, _ = fitter.fit(
                            bootstrap_trajectory,
                            value_columns,
                            search_start,
                            search_end,
                        )
                        if bootstrap_fit is None:
                            continue
                        bootstrap_times.append(
                            bootstrap_fit.candidate_time_sec
                        )
                        bootstrap_scores.append(bootstrap_fit.score)
                        bootstrap_rows.append(
                            {
                                "WindowLength_sec": int(window_sec),
                                "RRThreshold": float(rr_threshold),
                                "Representation": representation,
                                "Endpoint": endpoint,
                                "TransitionType": direction,
                                "SearchWindow": search_name,
                                "Bootstrap": int(bootstrap_index),
                                **_fit_to_dict(bootstrap_fit),
                                "ExceedsDualPseudoThreshold": bool(
                                    bootstrap_fit.score >= dual_threshold
                                ),
                            }
                        )
                    direction_bootstrap_times[(direction, search_name)] = bootstrap_times
                    timing_low, timing_high = (
                        np.nanquantile(bootstrap_times, [0.025, 0.975])
                        if bootstrap_times
                        else (np.nan, np.nan)
                    )
                    score_low, score_high = (
                        np.nanquantile(bootstrap_scores, [0.025, 0.975])
                        if bootstrap_scores
                        else (np.nan, np.nan)
                    )
                    mode_time, mode_probability = _mode_and_probability(
                        bootstrap_times
                    )
                    bootstrap_supported_fraction = (
                        float(
                            np.mean(
                                np.asarray(bootstrap_scores, dtype=float)
                                >= dual_threshold
                            )
                        )
                        if bootstrap_scores
                        else np.nan
                    )

                    summary_rows.append(
                        {
                            "WindowLength_sec": int(window_sec),
                            "RRThreshold": float(rr_threshold),
                            "Representation": representation,
                            "Endpoint": endpoint,
                            "TransitionType": direction,
                            "SearchWindow": search_name,
                            **_fit_to_dict(real_fit),
                            "PseudoA_Draws": int(len(null_a)),
                            "PseudoNA_Draws": int(len(null_na)),
                            "PseudoA_PeakThreshold": threshold_a,
                            "PseudoNA_PeakThreshold": threshold_na,
                            "DualControlThreshold": dual_threshold,
                            "EmpiricalPValue_PseudoA": p_a,
                            "EmpiricalPValue_PseudoNA": p_na,
                            "DualControlPValue": float(
                                np.nanmax([p_a, p_na])
                            ),
                            "SupportedAgainstPseudoA": supported_a,
                            "SupportedAgainstPseudoNA": supported_na,
                            "ValidatedAgainstBothPseudoControls": supported_both,
                            "TimingStatus": status,
                            "CandidateTimeCI95_Lower_sec": float(timing_low),
                            "CandidateTimeCI95_Upper_sec": float(timing_high),
                            "BootstrapModeTime_sec": mode_time,
                            "BootstrapModeProbability": mode_probability,
                            "PeakScoreCI95_Lower": float(score_low),
                            "PeakScoreCI95_Upper": float(score_high),
                            "BootstrapExceedsDualThresholdFraction": (
                                bootstrap_supported_fraction
                            ),
                            "Interpretation": (
                                "Validated shared timing"
                                if supported_both
                                else (
                                    "Candidate timing only; score did not exceed both "
                                    "stable-session pseudo-control nulls"
                                )
                            ),
                        }
                    )

                    if config.timing_lopo:
                        participants = sorted(
                            real_trajectory["Subject"].astype(str).unique()
                        )
                        for omitted in participants:
                            subset = real_trajectory[
                                ~real_trajectory["Subject"].astype(str).eq(omitted)
                            ]
                            lopo_fit, _ = fitter.fit(
                                subset,
                                value_columns,
                                search_start,
                                search_end,
                            )
                            lopo_rows.append(
                                {
                                    "WindowLength_sec": int(window_sec),
                                    "RRThreshold": float(rr_threshold),
                                    "Representation": representation,
                                    "Endpoint": endpoint,
                                    "TransitionType": direction,
                                    "SearchWindow": search_name,
                                    "OmittedSubject": omitted,
                                    **_fit_to_dict(lopo_fit),
                                }
                            )

            # Paired participant bootstrap timing difference for each search.
            common_subjects = sorted(
                set(
                    direction_trajectories.get("NA_to_A", pd.DataFrame())
                    .get("Subject", pd.Series(dtype=str))
                    .astype(str)
                    .unique()
                )
                & set(
                    direction_trajectories.get("A_to_NA", pd.DataFrame())
                    .get("Subject", pd.Series(dtype=str))
                    .astype(str)
                    .unique()
                )
            )
            if common_subjects:
                for search_name in config.timing_search_windows:
                    search_start, search_end = SEARCH_WINDOWS[search_name]
                    observed_na = direction_fits.get(("NA_to_A", search_name))
                    observed_a = direction_fits.get(("A_to_NA", search_name))
                    observed_difference = (
                        observed_a.candidate_time_sec
                        - observed_na.candidate_time_sec
                        if observed_a is not None and observed_na is not None
                        else np.nan
                    )
                    difference_draws: list[float] = []
                    common = np.asarray(common_subjects, dtype=object)
                    group_lookup: dict[str, dict[str, pd.DataFrame]] = {}
                    for direction in ["NA_to_A", "A_to_NA"]:
                        source = direction_trajectories[direction]
                        group_lookup[direction] = {
                            subject: source[
                                source["Subject"].astype(str).eq(subject)
                            ].copy()
                            for subject in common_subjects
                        }
                    for bootstrap_index in range(
                        config.timing_bootstrap_replicates
                    ):
                        sampled = rng.choice(
                            common,
                            size=len(common),
                            replace=True,
                        )
                        fit_by_direction: dict[str, TimingFit | None] = {}
                        for direction in ["NA_to_A", "A_to_NA"]:
                            parts: list[pd.DataFrame] = []
                            for copy_index, subject in enumerate(sampled):
                                part = group_lookup[direction][str(subject)].copy()
                                part["Subject"] = (
                                    f"{subject}__paired_bootstrap_{copy_index}"
                                )
                                parts.append(part)
                            bootstrap_source = pd.concat(parts, ignore_index=True)
                            fit_by_direction[direction], _ = fitter.fit(
                                bootstrap_source,
                                value_columns,
                                search_start,
                                search_end,
                            )
                        fit_na = fit_by_direction["NA_to_A"]
                        fit_a = fit_by_direction["A_to_NA"]
                        if fit_na is None or fit_a is None:
                            continue
                        difference_draws.append(
                            fit_a.candidate_time_sec - fit_na.candidate_time_sec
                        )
                    difference_low, difference_high = (
                        np.nanquantile(difference_draws, [0.025, 0.975])
                        if difference_draws
                        else (np.nan, np.nan)
                    )
                    direction_rows.append(
                        {
                            "WindowLength_sec": int(window_sec),
                            "RRThreshold": float(rr_threshold),
                            "Representation": representation,
                            "Endpoint": endpoint,
                            "SearchWindow": search_name,
                            "Contrast": "A_to_NA minus NA_to_A candidate time",
                            "NA_to_A_CandidateTime_sec": (
                                observed_na.candidate_time_sec
                                if observed_na is not None
                                else np.nan
                            ),
                            "A_to_NA_CandidateTime_sec": (
                                observed_a.candidate_time_sec
                                if observed_a is not None
                                else np.nan
                            ),
                            "Difference_sec": observed_difference,
                            "DifferenceCI95_Lower_sec": float(difference_low),
                            "DifferenceCI95_Upper_sec": float(difference_high),
                            "ParticipantsWithBothDirections": len(common_subjects),
                            "ComparisonStatus": (
                                "validated_only_if_both_direction_timings_pass_both_pseudo_controls"
                            ),
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    profiles = (
        pd.concat(real_profile_rows, ignore_index=True)
        if real_profile_rows
        else pd.DataFrame()
    )
    pseudo_null = pd.DataFrame(pseudo_null_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    lopo = pd.DataFrame(lopo_rows)
    direction_comparison = pd.DataFrame(direction_rows)
    simulation_raw = pd.DataFrame(simulation_rows)
    if simulation_raw.empty:
        simulation_power = simulation_raw
    else:
        simulation_power = (
            simulation_raw.groupby(
                [
                    "WindowLength_sec",
                    "RRThreshold",
                    "Representation",
                    "Endpoint",
                    "TransitionType",
                    "PseudoBaselineCondition",
                    "SearchWindow",
                    "EffectSizeSD",
                    "AffectedFraction",
                    "ChangeShape",
                    "ChangePosition",
                    "TrueTime_sec",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                Simulations=("Simulation", "size"),
                DetectionPower=(
                    "DetectedAgainstDualPseudoThreshold",
                    "mean",
                ),
                MedianEstimatedTime_sec=("EstimatedTime_sec", "median"),
                MedianAbsoluteTimingError_sec=(
                    "AbsoluteTimingError_sec",
                    "median",
                ),
                MeanAbsoluteTimingError_sec=(
                    "AbsoluteTimingError_sec",
                    "mean",
                ),
            )
        )
    return (
        summary,
        profiles,
        pseudo_null,
        bootstrap,
        lopo,
        direction_comparison,
        simulation_power,
    )
