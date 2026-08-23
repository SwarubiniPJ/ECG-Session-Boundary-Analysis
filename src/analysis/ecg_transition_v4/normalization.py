from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ALL23, NormalizationParameters
from .utils import robust_center_scale, stable_hash, transform_feature_values


def quality_mask(data: pd.DataFrame, rr_threshold: float) -> pd.Series:
    feature_complete = data[ALL23].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return (
        data["WindowValid"].fillna(False).astype(bool)
        & data["RR_CorrectedPercent"].le(float(rr_threshold))
        & feature_complete
    )


def stable_rows(
    data: pd.DataFrame,
    rr_threshold: float,
    stable_edge_sec: float,
) -> pd.DataFrame:
    mask = quality_mask(data, rr_threshold)
    mask &= data["StartTime_sec"].ge(stable_edge_sec)
    mask &= data["EndTime_sec"].le(data["SessionDuration_sec"] - stable_edge_sec)
    return data.loc[mask].copy()


def _deterministic_equal_sample(values: np.ndarray, n: int, key: str) -> np.ndarray:
    values = values[np.isfinite(values)]
    if n <= 0 or values.size == 0:
        return np.empty(0, dtype=float)
    if values.size <= n:
        return values.copy()
    # Stable pseudo-random order prevents always selecting the earliest windows.
    rng = np.random.default_rng(stable_hash(key) % (2**32 - 1))
    indices = rng.choice(np.arange(values.size), size=n, replace=False)
    return values[np.sort(indices)]


class SymmetricNormalizationStore:
    """Participant-specific, condition-balanced, session-excluding normalization.

    For each analysed real or pseudo boundary, normalization is fitted to stable
    A and NA observations from the same participant while excluding every session
    contributing observations to that boundary. Stable A and NA rows receive equal
    weight whenever both are available. The same rule is used for real and pseudo
    boundaries.
    """

    def __init__(
        self,
        window_data: pd.DataFrame,
        rr_threshold: float,
        stable_edge_sec: float,
        seed: int,
    ) -> None:
        self.data = window_data.copy()
        self.rr_threshold = float(rr_threshold)
        self.stable_edge_sec = float(stable_edge_sec)
        self.seed = int(seed)
        self.quality = self.data.loc[quality_mask(self.data, rr_threshold)].copy()
        self.stable = stable_rows(self.data, rr_threshold, stable_edge_sec)
        self.cache: dict[tuple[str, tuple[int, ...]], NormalizationParameters] = {}
        self.audit_rows: list[dict[str, object]] = []

    def _source(
        self, subject: str, excluded_sessions: tuple[int, ...]
    ) -> tuple[pd.DataFrame, str]:
        subject_stable = self.stable[self.stable["Subject"].astype(str).eq(subject)]
        source = subject_stable[~subject_stable["Session"].isin(excluded_sessions)].copy()
        if (
            source[source["ConditionCode"].eq("A")].shape[0] >= 5
            and source[source["ConditionCode"].eq("NA")].shape[0] >= 5
        ):
            return source, "stable_other_sessions_balanced_A_NA"

        subject_quality = self.quality[self.quality["Subject"].astype(str).eq(subject)]
        source = subject_quality[~subject_quality["Session"].isin(excluded_sessions)].copy()
        if (
            source[source["ConditionCode"].eq("A")].shape[0] >= 5
            and source[source["ConditionCode"].eq("NA")].shape[0] >= 5
        ):
            return source, "all_quality_other_sessions_balanced_A_NA"

        # Never reintroduce the analysed session(s). If condition balancing is
        # impossible after exclusion, retain an unbalanced other-session source
        # and record that fallback explicitly.
        if len(source) >= 10:
            return source, "all_quality_other_sessions_unbalanced_fallback"
        if len(subject_stable[~subject_stable["Session"].isin(excluded_sessions)]) >= 10:
            return (
                subject_stable[~subject_stable["Session"].isin(excluded_sessions)].copy(),
                "stable_other_sessions_unbalanced_fallback",
            )
        raise RuntimeError(
            f"Insufficient normalization observations for subject {subject}, "
            f"excluding sessions {excluded_sessions}; analysed sessions were not reused."
        )

    def fit(
        self, subject: str, excluded_sessions: Iterable[int]
    ) -> NormalizationParameters:
        excluded = tuple(sorted({int(value) for value in excluded_sessions}))
        key = (str(subject), excluded)
        if key in self.cache:
            return self.cache[key]

        source, source_name = self._source(str(subject), excluded)
        group_a = source[source["ConditionCode"].astype(str).str.upper().eq("A")]
        group_na = source[source["ConditionCode"].astype(str).str.upper().eq("NA")]
        centers: dict[str, float] = {}
        scales: dict[str, float] = {}
        methods: dict[str, str] = {}

        for feature in ALL23:
            values_a = transform_feature_values(group_a[feature], feature)
            values_na = transform_feature_values(group_na[feature], feature)
            finite_a = values_a[np.isfinite(values_a)]
            finite_na = values_na[np.isfinite(values_na)]
            if finite_a.size >= 3 and finite_na.size >= 3:
                n = int(min(finite_a.size, finite_na.size))
                sample_a = _deterministic_equal_sample(
                    finite_a, n, f"{self.seed}|{subject}|{excluded}|{feature}|A"
                )
                sample_na = _deterministic_equal_sample(
                    finite_na, n, f"{self.seed}|{subject}|{excluded}|{feature}|NA"
                )
                balanced = np.concatenate([sample_a, sample_na])
            else:
                balanced = transform_feature_values(source[feature], feature)
                balanced = balanced[np.isfinite(balanced)]
            center, scale, method = robust_center_scale(balanced)
            if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
                raise RuntimeError(
                    f"Invalid normalization for {subject}, {feature}, excluded={excluded}."
                )
            centers[feature] = center
            scales[feature] = scale
            methods[feature] = method

        params = NormalizationParameters(
            subject=str(subject),
            excluded_sessions=excluded,
            rows_a=int(len(group_a)),
            rows_na=int(len(group_na)),
            source=source_name,
            centers=centers,
            scales=scales,
            scale_methods=methods,
        )
        self.cache[key] = params
        audit = asdict(params)
        audit.pop("centers")
        audit.pop("scales")
        audit.pop("scale_methods")
        self.audit_rows.append(audit)
        return params

    def apply(
        self,
        frame: pd.DataFrame,
        subject: str,
        excluded_sessions: Iterable[int],
    ) -> pd.DataFrame:
        params = self.fit(subject, excluded_sessions)
        out = frame.copy()
        for feature in ALL23:
            transformed = transform_feature_values(out[feature], feature)
            out[f"{feature}_z"] = (
                transformed - params.centers[feature]
            ) / params.scales[feature]
        out["NormalizationSource"] = params.source
        out["NormalizationExcludedSessions"] = ";".join(
            str(value) for value in params.excluded_sessions
        )
        return out

    def audit_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.audit_rows).drop_duplicates()


def crossfit_stable_normalization(
    store: SymmetricNormalizationStore,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for (subject, session), group in store.stable.groupby(
        ["Subject", "Session"], sort=True
    ):
        normalized = store.apply(group, str(subject), [int(session)])
        normalized["StableCrossfitExcludedSession"] = int(session)
        pieces.append(normalized)
    if not pieces:
        raise RuntimeError("No stable observations were available for cross-fitted normalization.")
    return pd.concat(pieces, ignore_index=True)
