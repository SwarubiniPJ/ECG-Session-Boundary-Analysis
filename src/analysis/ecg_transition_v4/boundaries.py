from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ALL23, NEUTRAL_DIRECTION_LABELS
from .normalization import SymmetricNormalizationStore, quality_mask
from .utils import stable_hash


def construct_real_boundary_observations(
    data: pd.DataFrame,
    inventory: pd.DataFrame,
    normalizer: SymmetricNormalizationStore,
    rr_threshold: float,
    pre_sec: float,
    post_sec: float,
    min_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = data.loc[quality_mask(data, rr_threshold)].copy()
    pieces: list[pd.DataFrame] = []
    eligibility_rows: list[dict[str, object]] = []
    for boundary in inventory.itertuples(index=False):
        pre = selected[
            selected["Subject"].astype(str).eq(str(boundary.Subject))
            & selected["Session"].eq(int(boundary.PreSession))
        ].copy()
        post = selected[
            selected["Subject"].astype(str).eq(str(boundary.Subject))
            & selected["Session"].eq(int(boundary.PostSession))
        ].copy()
        pre["RelativeStart_sec"] = pre["StartTime_sec"] - pre["SessionDuration_sec"]
        pre["RelativeEnd_sec"] = pre["EndTime_sec"] - pre["SessionDuration_sec"]
        pre["RelativeCenter_sec"] = pre["CenterTime_sec"] - pre["SessionDuration_sec"]
        pre = pre[pre["RelativeCenter_sec"].ge(-pre_sec)]
        pre["Side"] = "Pre"
        post["RelativeStart_sec"] = post["StartTime_sec"]
        post["RelativeEnd_sec"] = post["EndTime_sec"]
        post["RelativeCenter_sec"] = post["CenterTime_sec"]
        post = post[post["RelativeCenter_sec"].le(post_sec)]
        post["Side"] = "Post"
        eligible = len(pre) >= min_points and len(post) >= min_points
        eligibility_rows.append(
            {
                "BoundaryID": boundary.BoundaryID,
                "Subject": boundary.Subject,
                "TransitionType": boundary.TransitionType,
                "BoundaryOrder": boundary.BoundaryOrder,
                "RRThreshold": rr_threshold,
                "PreCount": len(pre),
                "PostCount": len(post),
                "Eligible": eligible,
            }
        )
        if not eligible:
            continue
        combined = pd.concat([pre, post], ignore_index=True)
        combined = normalizer.apply(
            combined,
            str(boundary.Subject),
            [int(boundary.PreSession), int(boundary.PostSession)],
        )
        combined["BoundaryID"] = boundary.BoundaryID
        combined["BoundaryKind"] = "Real"
        combined["Subject"] = str(boundary.Subject)
        combined["PreSession"] = int(boundary.PreSession)
        combined["PostSession"] = int(boundary.PostSession)
        combined["BoundaryOrder"] = int(boundary.BoundaryOrder)
        combined["TransitionType"] = boundary.TransitionType
        combined["TransitionLabel"] = NEUTRAL_DIRECTION_LABELS.get(
            boundary.TransitionType, boundary.TransitionType
        )
        combined["PseudoCondition"] = ""
        combined["CandidateID"] = ""
        combined["CandidateBlockID"] = ""
        combined["PseudoFold"] = -1
        combined["RRThreshold"] = float(rr_threshold)
        pieces.append(combined)
    return (
        pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(),
        pd.DataFrame(eligibility_rows),
    )


def candidate_pseudo_boundaries_for_session(
    session_rows: pd.DataFrame,
    subject: str,
    session: int,
    condition: str,
    rr_threshold: float,
    pre_sec: float,
    post_sec: float,
    min_points: int,
    block_separation_sec: float,
) -> list[dict[str, object]]:
    """Return non-overlapping, quality-eligible within-session pseudo controls.

    A pseudo sequence retains window centres from ``-pre_sec`` to ``+post_sec``.
    Because each centre represents a finite window, the underlying raw-signal
    support is wider by half a window on each side. Candidate pseudo times are
    therefore thinned so that their complete raw-signal supports do not overlap.
    """
    rows = session_rows.loc[quality_mask(session_rows, rr_threshold)].copy()
    if rows.empty:
        return []
    starts = set(np.round(rows["StartTime_sec"].to_numpy(dtype=float), 6))
    ends = set(np.round(rows["EndTime_sec"].to_numpy(dtype=float), 6))
    candidate_times = sorted(starts.intersection(ends))
    dense: list[dict[str, object]] = []
    duration = float(rows["SessionDuration_sec"].iloc[0])
    window_sec = float(rows["WindowLength_sec"].iloc[0])
    raw_support_separation = float(pre_sec + post_sec + window_sec)
    independence_gap = max(float(block_separation_sec), raw_support_separation)

    for t0 in candidate_times:
        pre = rows[
            rows["EndTime_sec"].le(t0 + 1e-6)
            & rows["CenterTime_sec"].ge(t0 - pre_sec - 1e-6)
        ]
        post = rows[
            rows["StartTime_sec"].ge(t0 - 1e-6)
            & rows["CenterTime_sec"].le(t0 + post_sec + 1e-6)
        ]
        if len(pre) < min_points or len(post) < min_points:
            continue
        dense.append(
            {
                "Subject": str(subject),
                "Session": int(session),
                "ConditionCode": str(condition).upper(),
                "PseudoTime_sec": float(t0),
                "SessionDuration_sec": duration,
                "WindowLength_sec": window_sec,
                "PreCount": int(len(pre)),
                "PostCount": int(len(post)),
            }
        )
    if not dense:
        return []

    # Greedy thinning over the sorted time grid guarantees that retained pseudo
    # sequences have disjoint raw-signal support. The same retained candidate is
    # used wherever its block is matched, and the whole block is assigned to one
    # cross-fitting fold.
    independent: list[dict[str, object]] = []
    last_time = -np.inf
    for row in dense:
        t0 = float(row["PseudoTime_sec"])
        if t0 < last_time + independence_gap - 1e-6:
            continue
        independent.append(row)
        last_time = t0

    for block_index, row in enumerate(independent):
        t0 = float(row["PseudoTime_sec"])
        block_id = f"{subject}_S{session}_{condition}_BLK{block_index:03d}"
        candidate_id = f"{subject}_S{session}_{condition}_T{t0:g}"
        row["CandidateID"] = candidate_id
        row["CandidateBlockID"] = block_id
        row["BlockIndex"] = block_index
        row["BlockAnchorTime_sec"] = t0
        row["RelativeSessionPosition"] = t0 / max(duration, 1e-12)
        row["RawSupportStart_sec"] = t0 - pre_sec - window_sec / 2.0
        row["RawSupportEnd_sec"] = t0 + post_sec + window_sec / 2.0
        row["IndependenceGap_sec"] = independence_gap
        row["DenseCandidatesBeforeThinning"] = len(dense)
        row["IndependentCandidatesAfterThinning"] = len(independent)
    return independent


def build_pseudo_candidate_table(
    data: pd.DataFrame,
    rr_threshold: float,
    pre_sec: float,
    post_sec: float,
    min_points: int,
    block_separation_sec: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subject, session, condition), group in data.groupby(
        ["Subject", "Session", "ConditionCode"], sort=True
    ):
        rows.extend(
            candidate_pseudo_boundaries_for_session(
                group,
                str(subject),
                int(session),
                str(condition),
                rr_threshold,
                pre_sec,
                post_sec,
                min_points,
                block_separation_sec,
            )
        )
    return pd.DataFrame(rows)


def _choose_best_candidate_per_block(
    pool: pd.DataFrame,
    target_pre_count: int,
    target_post_count: int,
    target_session: int,
    target_position: float,
) -> pd.DataFrame:
    if pool.empty:
        return pool
    scored = pool.copy()
    scored["CountMismatch"] = (
        (scored["PreCount"] - target_pre_count).abs()
        + (scored["PostCount"] - target_post_count).abs()
    )
    scored["SessionDistance"] = (scored["Session"] - target_session).abs()
    scored["PositionDistance"] = (
        scored["RelativeSessionPosition"] - target_position
    ).abs()
    scored["MatchScore"] = (
        100.0 * scored["CountMismatch"]
        + 3.0 * scored["SessionDistance"]
        + scored["PositionDistance"]
    )
    scored = scored.sort_values(
        ["CandidateBlockID", "MatchScore", "CandidateID"], kind="stable"
    )
    return scored.groupby("CandidateBlockID", as_index=False).first()


def match_pseudo_controls(
    real_inventory: pd.DataFrame,
    real_eligibility: pd.DataFrame,
    candidates: pd.DataFrame,
    controls_per_boundary: int,
    crossfit_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = real_eligibility[real_eligibility["Eligible"]].merge(
        real_inventory,
        on=["BoundaryID", "Subject", "TransitionType", "BoundaryOrder"],
        how="left",
        validate="one_to_one",
    )
    match_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    # Assign whole non-overlapping pseudo blocks to folds once. Sequential
    # allocation within participant and condition keeps folds balanced while
    # guaranteeing that a block can never appear in both calibration and
    # held-out evaluation, even when it is matched to several real boundaries.
    block_fold_map: dict[str, int] = {}
    unique_blocks = (
        candidates.sort_values(["Subject", "ConditionCode", "Session", "BlockIndex"])
        .drop_duplicates("CandidateBlockID")
    )
    for (subject, condition), group in unique_blocks.groupby(
        ["Subject", "ConditionCode"], sort=True
    ):
        offset = stable_hash(f"{subject}|{condition}", seed) % crossfit_folds
        for position, block_id in enumerate(group["CandidateBlockID"].astype(str)):
            block_fold_map[block_id] = int((position + offset) % crossfit_folds)

    for real in eligible.sort_values(["Subject", "BoundaryOrder"]).itertuples(index=False):
        for pseudo_condition in ["A", "NA"]:
            target_session = (
                int(real.PreSession)
                if str(real.PreConditionCode) == pseudo_condition
                else int(real.PostSession)
            )
            target_duration = (
                float(real.PreDuration_sec)
                if str(real.PreConditionCode) == pseudo_condition
                else float(real.PostDuration_sec)
            )
            # A central within-session pseudo point is the neutral matching target.
            target_position = 0.5
            pool = candidates[
                candidates["Subject"].astype(str).eq(str(real.Subject))
                & candidates["ConditionCode"].eq(pseudo_condition)
            ].copy()
            best = _choose_best_candidate_per_block(
                pool,
                int(real.PreCount),
                int(real.PostCount),
                target_session,
                target_position,
            )
            best = best.sort_values(
                ["MatchScore", "Session", "PseudoTime_sec"], kind="stable"
            ).head(int(controls_per_boundary))
            audit_rows.append(
                {
                    "MatchedRealBoundaryID": real.BoundaryID,
                    "Subject": real.Subject,
                    "TransitionType": real.TransitionType,
                    "PseudoCondition": pseudo_condition,
                    "AvailableIndependentCandidates": int(len(pool)),
                    "AvailableIndependentBlocks": int(pool["CandidateBlockID"].nunique()),
                    "DenseCandidatesBeforeThinning": (
                        int(pool["DenseCandidatesBeforeThinning"].max())
                        if len(pool) and "DenseCandidatesBeforeThinning" in pool.columns
                        else 0
                    ),
                    "SelectedControls": int(len(best)),
                    "RequestedControls": int(controls_per_boundary),
                }
            )
            for rank, candidate in enumerate(best.itertuples(index=False), start=1):
                fold = block_fold_map[str(candidate.CandidateBlockID)]
                match_rows.append(
                    {
                        "PseudoMatchID": (
                            f"{real.BoundaryID}__{pseudo_condition}__{candidate.CandidateID}"
                        ),
                        "MatchedRealBoundaryID": real.BoundaryID,
                        "Subject": real.Subject,
                        "TransitionType": real.TransitionType,
                        "TransitionLabel": real.TransitionLabel,
                        "BoundaryOrder": int(real.BoundaryOrder),
                        "PseudoCondition": pseudo_condition,
                        "CandidateID": candidate.CandidateID,
                        "CandidateBlockID": candidate.CandidateBlockID,
                        "PseudoFold": int(fold),
                        "PseudoSession": int(candidate.Session),
                        "PseudoTime_sec": float(candidate.PseudoTime_sec),
                        "TargetSession": target_session,
                        "TargetDuration_sec": target_duration,
                        "TargetPreCount": int(real.PreCount),
                        "TargetPostCount": int(real.PostCount),
                        "PseudoPreCount": int(candidate.PreCount),
                        "PseudoPostCount": int(candidate.PostCount),
                        "CountMismatch": int(candidate.CountMismatch),
                        "SessionDistance": int(candidate.SessionDistance),
                        "PositionDistance": float(candidate.PositionDistance),
                        "MatchScore": float(candidate.MatchScore),
                        "MatchRank": int(rank),
                    }
                )
    return pd.DataFrame(match_rows), pd.DataFrame(audit_rows)


def construct_pseudo_boundary_observations(
    data: pd.DataFrame,
    matches: pd.DataFrame,
    candidates: pd.DataFrame,
    normalizer: SymmetricNormalizationStore,
    rr_threshold: float,
    pre_sec: float,
    post_sec: float,
    min_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matches.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected = data.loc[quality_mask(data, rr_threshold)].copy()
    candidate_lookup = (
        candidates[candidates["CandidateID"].isin(matches["CandidateID"].unique())]
        .drop_duplicates("CandidateID")
        .set_index("CandidateID")
    )
    fold_lookup = matches.groupby("CandidateID")["PseudoFold"].first()
    pieces: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for candidate_id, candidate in candidate_lookup.iterrows():
        session_rows = selected[
            selected["Subject"].astype(str).eq(str(candidate.Subject))
            & selected["Session"].eq(int(candidate.Session))
        ].copy()
        t0 = float(candidate.PseudoTime_sec)
        pre = session_rows[
            session_rows["EndTime_sec"].le(t0 + 1e-6)
            & session_rows["CenterTime_sec"].ge(t0 - pre_sec - 1e-6)
        ].copy()
        post = session_rows[
            session_rows["StartTime_sec"].ge(t0 - 1e-6)
            & session_rows["CenterTime_sec"].le(t0 + post_sec + 1e-6)
        ].copy()
        eligible = len(pre) >= min_points and len(post) >= min_points
        audit_rows.append(
            {
                "CandidateID": candidate_id,
                "CandidateBlockID": candidate.CandidateBlockID,
                "Subject": candidate.Subject,
                "Session": candidate.Session,
                "ConditionCode": candidate.ConditionCode,
                "PseudoTime_sec": t0,
                "PreCount": len(pre),
                "PostCount": len(post),
                "Eligible": eligible,
                "PseudoFold": int(fold_lookup.get(candidate_id, -1)),
            }
        )
        if not eligible:
            continue
        pre["RelativeStart_sec"] = pre["StartTime_sec"] - t0
        pre["RelativeEnd_sec"] = pre["EndTime_sec"] - t0
        pre["RelativeCenter_sec"] = pre["CenterTime_sec"] - t0
        pre["Side"] = "Pre"
        post["RelativeStart_sec"] = post["StartTime_sec"] - t0
        post["RelativeEnd_sec"] = post["EndTime_sec"] - t0
        post["RelativeCenter_sec"] = post["CenterTime_sec"] - t0
        post["Side"] = "Post"
        combined = pd.concat([pre, post], ignore_index=True)
        combined = normalizer.apply(
            combined,
            str(candidate.Subject),
            [int(candidate.Session)],
        )
        combined["BoundaryID"] = candidate_id
        combined["BoundaryKind"] = "Pseudo"
        combined["Subject"] = str(candidate.Subject)
        combined["PseudoCondition"] = str(candidate.ConditionCode)
        combined["CandidateID"] = candidate_id
        combined["CandidateBlockID"] = str(candidate.CandidateBlockID)
        combined["PseudoFold"] = int(fold_lookup.get(candidate_id, -1))
        combined["PreSession"] = int(candidate.Session)
        combined["PostSession"] = int(candidate.Session)
        combined["BoundaryOrder"] = -1
        combined["TransitionType"] = ""
        combined["TransitionLabel"] = "Stable within-session pseudo-boundary"
        combined["RRThreshold"] = float(rr_threshold)
        pieces.append(combined)
    return (
        pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(),
        pd.DataFrame(audit_rows),
    )
