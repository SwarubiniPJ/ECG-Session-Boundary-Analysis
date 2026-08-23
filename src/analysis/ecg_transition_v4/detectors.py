from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import RepresentationSpec, SEARCH_WINDOWS
from .utils import effective_sample_size

try:
    import ruptures as rpt  # type: ignore
except Exception:  # pragma: no cover
    rpt = None


def pre_standardize_matrix(times: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(matrix, dtype=float).copy()
    pre = times < 0
    if pre.sum() < 2:
        return out
    for index in range(out.shape[1]):
        center = float(np.median(out[pre, index]))
        mad = float(np.median(np.abs(out[pre, index] - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(out[pre, index], ddof=1))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        out[:, index] = (out[:, index] - center) / scale
    return out


def candidate_splits(
    times: NDArray[np.float64],
    min_points: int,
    search_start_sec: float,
    search_end_sec: float,
) -> list[int]:
    n = len(times)
    return [
        split
        for split in range(min_points, n - min_points + 1)
        if split < n
        and float(times[split]) >= search_start_sec
        and float(times[split]) <= search_end_sec
    ]


def bic_from_sse(sse: float, n: int, k: int) -> float:
    safe = max(float(sse), np.finfo(float).tiny)
    count = max(int(n), 1)
    return count * math.log(safe / count) + k * math.log(count)


def mahalanobis_segment_cost(
    matrix: NDArray[np.float64], inverse_covariance: NDArray[np.float64]
) -> float:
    if len(matrix) == 0:
        return np.inf
    residuals = matrix - np.mean(matrix, axis=0, keepdims=True)
    return float(np.einsum("ij,jk,ik->", residuals, inverse_covariance, residuals))


def legacy_bic_score(
    times: NDArray[np.float64],
    matrix: NDArray[np.float64],
    min_points: int,
    search_start_sec: float,
    search_end_sec: float,
) -> dict[str, object]:
    x = pre_standardize_matrix(times, matrix)
    n, d = x.shape
    overall = np.mean(x, axis=0, keepdims=True)
    sse0 = float(np.sum((x - overall) ** 2))
    bic0 = bic_from_sse(sse0, n * d, d)
    best: tuple[float, int, float] | None = None
    for split in candidate_splits(times, min_points, search_start_sec, search_end_sec):
        left, right = x[:split], x[split:]
        sse1 = float(
            np.sum((left - np.mean(left, axis=0, keepdims=True)) ** 2)
            + np.sum((right - np.mean(right, axis=0, keepdims=True)) ** 2)
        )
        bic1 = bic_from_sse(sse1, n * d, 2 * d + 1)
        score = bic0 - bic1
        if best is None or score > best[0]:
            best = (score, split, sse1)
    if best is None:
        return {"Score": np.nan, "SplitIndex": -1, "CandidateLatency_sec": np.nan}
    score, split, sse1 = best
    return {
        "Score": float(score),
        "SplitIndex": int(split),
        "CandidateLatency_sec": float(times[split]),
        "SSE_NoChange": sse0,
        "SSE_OneChange": sse1,
        "BIC_NoChange": bic0,
        "BIC_OneChange": bic0 - score,
    }


def covariance_ic_score(
    times: NDArray[np.float64],
    matrix: NDArray[np.float64],
    spec: RepresentationSpec,
    min_points: int,
    search_start_sec: float,
    search_end_sec: float,
) -> dict[str, object]:
    x = np.asarray(matrix, dtype=float)
    n, d = x.shape
    q0 = mahalanobis_segment_cost(x, spec.inverse_covariance)
    neff = effective_sample_size(n, spec.lag1_rho)
    penalty = (d + 1) * math.log(neff)
    best: tuple[float, int, float, float] | None = None
    for split in candidate_splits(times, min_points, search_start_sec, search_end_sec):
        q1 = mahalanobis_segment_cost(x[:split], spec.inverse_covariance) + mahalanobis_segment_cost(
            x[split:], spec.inverse_covariance
        )
        gain = q0 - q1
        score = gain - penalty
        if best is None or score > best[0]:
            best = (score, split, q1, gain)
    if best is None:
        return {"Score": np.nan, "SplitIndex": -1, "CandidateLatency_sec": np.nan}
    score, split, q1, gain = best
    return {
        "Score": float(score),
        "SplitIndex": int(split),
        "CandidateLatency_sec": float(times[split]),
        "Q_NoChange": q0,
        "Q_OneChange": q1,
        "LikelihoodGain": gain,
        "ParameterPenalty": penalty,
        "EffectiveSampleSize": neff,
    }


def binseg_l2_score(
    times: NDArray[np.float64],
    matrix: NDArray[np.float64],
    min_points: int,
    search_start_sec: float,
    search_end_sec: float,
) -> dict[str, object]:
    x = pre_standardize_matrix(times, matrix)
    overall = np.mean(x, axis=0, keepdims=True)
    cost0 = float(np.sum((x - overall) ** 2))
    best: tuple[float, int, float] | None = None
    for split in candidate_splits(times, min_points, search_start_sec, search_end_sec):
        left, right = x[:split], x[split:]
        cost1 = float(
            np.sum((left - np.mean(left, axis=0, keepdims=True)) ** 2)
            + np.sum((right - np.mean(right, axis=0, keepdims=True)) ** 2)
        )
        gain = cost0 - cost1
        if best is None or gain > best[0]:
            best = (gain, split, cost1)
    if best is None:
        return {"Score": np.nan, "SplitIndex": -1, "CandidateLatency_sec": np.nan}
    gain, split, cost1 = best
    return {
        "Score": float(gain),
        "SplitIndex": int(split),
        "CandidateLatency_sec": float(times[split]),
        "Cost_NoChange": cost0,
        "Cost_OneChange": cost1,
    }


def _multivariate_ols_sse(y: NDArray[np.float64], design: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    return float(np.sum(residuals**2)), beta


def segmented_trend_score(
    times: NDArray[np.float64],
    matrix: NDArray[np.float64],
    spec: RepresentationSpec,
    min_points: int,
    search_start_sec: float,
    search_end_sec: float,
) -> dict[str, object]:
    x = np.asarray(matrix, dtype=float) @ spec.whitening
    t = np.asarray(times, dtype=float) / 60.0
    base_design = np.column_stack([np.ones(len(t)), t])
    sse0, beta0 = _multivariate_ols_sse(x, base_design)
    neff = effective_sample_size(len(t), spec.lag1_rho)
    d = x.shape[1]
    penalty = (2 * d + 1) * math.log(neff)
    best: tuple[float, int, float, NDArray[np.float64], float] | None = None
    for split in candidate_splits(times, min_points, search_start_sec, search_end_sec):
        tau = float(t[split])
        step = (t >= tau).astype(float)
        hinge = np.maximum(t - tau, 0.0)
        design = np.column_stack([np.ones(len(t)), t, step, hinge])
        sse1, beta = _multivariate_ols_sse(x, design)
        gain = sse0 - sse1
        score = gain - penalty
        if best is None or score > best[0]:
            best = (score, split, sse1, beta, gain)
    if best is None:
        return {"Score": np.nan, "SplitIndex": -1, "CandidateLatency_sec": np.nan}
    score, split, sse1, beta, gain = best
    level = beta[2]
    slope = beta[3]
    return {
        "Score": float(score),
        "SplitIndex": int(split),
        "CandidateLatency_sec": float(times[split]),
        "SSE_Linear": sse0,
        "SSE_Segmented": sse1,
        "LikelihoodGain": gain,
        "ParameterPenalty": penalty,
        "LevelChangeNorm": float(np.linalg.norm(level)),
        "SlopeChangeNorm": float(np.linalg.norm(slope)),
        "SignedFirstLevelChange": float(level[0]) if len(level) else np.nan,
        "SignedFirstSlopeChange": float(slope[0]) if len(slope) else np.nan,
    }


def cusum_score(
    times: NDArray[np.float64],
    matrix: NDArray[np.float64],
    spec: RepresentationSpec,
    min_points: int,
    search_start_sec: float,
    search_end_sec: float,
) -> dict[str, object]:
    x = np.asarray(matrix, dtype=float) @ spec.whitening
    pre = times < 0
    center = np.mean(x[pre], axis=0, keepdims=True) if pre.any() else np.mean(x, axis=0, keepdims=True)
    centered = x - center
    cumulative = np.cumsum(centered, axis=0)
    total = cumulative[-1]
    n = len(x)
    best: tuple[float, int, NDArray[np.float64]] | None = None
    for split in candidate_splits(times, min_points, search_start_sec, search_end_sec):
        bridge = cumulative[split - 1] - (split / n) * total
        denominator = math.sqrt(max(split * (n - split) / n, 1e-12))
        vector = bridge / denominator
        score = float(np.linalg.norm(vector))
        if best is None or score > best[0]:
            best = (score, split, vector)
    if best is None:
        return {"Score": np.nan, "SplitIndex": -1, "CandidateLatency_sec": np.nan}
    score, split, vector = best
    return {
        "Score": score,
        "SplitIndex": int(split),
        "CandidateLatency_sec": float(times[split]),
        "SignedFirstCUSUM": float(vector[0]) if len(vector) else np.nan,
    }


def mosum_score(
    times: NDArray[np.float64],
    matrix: NDArray[np.float64],
    spec: RepresentationSpec,
    min_points: int,
    search_start_sec: float,
    search_end_sec: float,
) -> dict[str, object]:
    x = np.asarray(matrix, dtype=float) @ spec.whitening
    h = int(min_points)
    best: tuple[float, int, NDArray[np.float64]] | None = None
    for split in candidate_splits(times, min_points, search_start_sec, search_end_sec):
        left = x[max(0, split - h):split]
        right = x[split:min(len(x), split + h)]
        if len(left) < h or len(right) < h:
            continue
        difference = np.mean(right, axis=0) - np.mean(left, axis=0)
        vector = math.sqrt(h / 2.0) * difference
        score = float(np.linalg.norm(vector))
        if best is None or score > best[0]:
            best = (score, split, vector)
    if best is None:
        return {"Score": np.nan, "SplitIndex": -1, "CandidateLatency_sec": np.nan}
    score, split, vector = best
    return {
        "Score": score,
        "SplitIndex": int(split),
        "CandidateLatency_sec": float(times[split]),
        "SignedFirstMOSUM": float(vector[0]) if len(vector) else np.nan,
    }


class SegmentCost:
    def __init__(self, matrix: NDArray[np.float64], model: str):
        self.x = np.asarray(matrix, dtype=float)
        self.model = str(model)
        self.n, self.d = self.x.shape
        if model == "l2":
            self.prefix = np.vstack([np.zeros((1, self.d)), np.cumsum(self.x, axis=0)])
            self.prefix_sq = np.vstack([np.zeros((1, self.d)), np.cumsum(self.x * self.x, axis=0)])
        elif model == "rbf":
            differences = self.x[:, None, :] - self.x[None, :, :]
            distances = np.sum(differences * differences, axis=2)
            nonzero = distances[distances > np.finfo(float).eps]
            median_distance = float(np.median(nonzero)) if nonzero.size else 1.0
            gamma = 1.0 / max(median_distance, np.finfo(float).eps)
            self.gram = np.exp(-gamma * distances)
            self.gram_prefix = np.pad(
                np.cumsum(np.cumsum(self.gram, axis=0), axis=1),
                ((1, 0), (1, 0)),
                mode="constant",
            )
        elif model == "l1":
            pass
        else:
            raise ValueError(f"Unsupported segment cost model: {model}")

    def cost(self, start: int, end: int) -> float:
        length = end - start
        if length <= 0:
            return np.inf
        if self.model == "l2":
            sums = self.prefix[end] - self.prefix[start]
            sums_sq = self.prefix_sq[end] - self.prefix_sq[start]
            return float(np.sum(sums_sq - sums * sums / length))
        if self.model == "l1":
            segment = self.x[start:end]
            median = np.median(segment, axis=0, keepdims=True)
            return float(np.sum(np.abs(segment - median)))
        block_sum = (
            self.gram_prefix[end, end]
            - self.gram_prefix[start, end]
            - self.gram_prefix[end, start]
            + self.gram_prefix[start, start]
        )
        diagonal = float(np.trace(self.gram[start:end, start:end]))
        return float(diagonal - block_sum / length)


def exact_penalized_segmentation_from_cost(
    cost: SegmentCost, penalty: float, min_size: int
) -> dict[str, object]:
    """Solve the additive PELT objective while reusing precomputed segment costs."""
    n = cost.n
    objective = np.full(n + 1, np.inf, dtype=float)
    objective[0] = -float(penalty)
    paths: list[list[int]] = [[] for _ in range(n + 1)]
    for end in range(min_size, n + 1):
        best_value = np.inf
        best_start = -1
        starts = [0] + list(range(min_size, end - min_size + 1))
        for start in starts:
            if start > 0 and not np.isfinite(objective[start]):
                continue
            if end - start < min_size:
                continue
            value = objective[start] + cost.cost(start, end) + penalty
            if value < best_value:
                best_value = value
                best_start = start
        if best_start >= 0:
            objective[end] = best_value
            paths[end] = paths[best_start] + ([best_start] if best_start > 0 else [])
    breakpoints = paths[n]
    segments = list(zip([0] + breakpoints, breakpoints + [n]))
    segmented_cost = float(sum(cost.cost(start, end) for start, end in segments))
    no_change_cost = float(cost.cost(0, n))
    return {
        "Breakpoints": breakpoints,
        "Objective": float(objective[n]),
        "SegmentedCost": segmented_cost,
        "NoChangeCost": no_change_cost,
        "CostGain": no_change_cost - segmented_cost,
        "Implementation": "exact_dynamic_programming_PELT_objective",
    }


def exact_penalized_segmentation(
    matrix: NDArray[np.float64], model: str, penalty: float, min_size: int
) -> dict[str, object]:
    return exact_penalized_segmentation_from_cost(
        SegmentCost(matrix, model), penalty, min_size
    )


def pelt_base_penalty(n: int, d: int, rho: float, model: str) -> float:
    neff = effective_sample_size(n, rho)
    return float((d + 1) * math.log(neff)) if model in {"l1", "l2"} else float(math.log(neff))


def pelt_result(
    times: NDArray[np.float64],
    matrix: NDArray[np.float64],
    spec: RepresentationSpec,
    model: str,
    multiplier: float,
    min_points: int,
    search_start_sec: float,
    search_end_sec: float,
    require_ruptures: bool = False,
) -> dict[str, object]:
    x = np.asarray(matrix, dtype=float) @ spec.whitening
    base = pelt_base_penalty(len(x), x.shape[1], spec.lag1_rho, model)
    penalty = float(multiplier * base)
    if rpt is not None:
        breakpoints_with_end = rpt.Pelt(model=model, min_size=min_points, jump=1).fit(x).predict(pen=penalty)
        breakpoints = [int(value) for value in breakpoints_with_end if int(value) < len(x)]
        cost = SegmentCost(x, model)
        segments = list(zip([0] + breakpoints, breakpoints + [len(x)]))
        segmented_cost = float(sum(cost.cost(start, end) for start, end in segments))
        no_change_cost = float(cost.cost(0, len(x)))
        implementation = "ruptures.Pelt"
    else:
        if require_ruptures:
            raise ImportError(
                "ruptures is required by --require-ruptures. Install with: pip install ruptures"
            )
        exact = exact_penalized_segmentation(x, model, penalty, min_points)
        breakpoints = [int(value) for value in exact["Breakpoints"]]
        segmented_cost = float(exact["SegmentedCost"])
        no_change_cost = float(exact["NoChangeCost"])
        implementation = str(exact["Implementation"])
    valid = [
        split for split in breakpoints
        if split < len(times)
        and float(times[split]) >= search_start_sec
        and float(times[split]) <= search_end_sec
    ]
    split = valid[0] if valid else -1
    return {
        "DetectedAtPenalty": bool(valid),
        "SplitIndex": int(split),
        "CandidateLatency_sec": float(times[split]) if split >= 0 else np.nan,
        "Score": float(no_change_cost - segmented_cost),
        "Penalty": penalty,
        "PenaltyMultiplier": float(multiplier),
        "BasePenalty": base,
        "NumBreakpoints": len(breakpoints),
        "Breakpoints": ";".join(str(value) for value in breakpoints),
        "NoChangeCost": no_change_cost,
        "SegmentedCost": segmented_cost,
        "Implementation": implementation,
    }


def fixed_boundary_magnitude(
    times: NDArray[np.float64],
    matrix: NDArray[np.float64],
    spec: RepresentationSpec,
) -> dict[str, float]:
    pre = matrix[times < 0]
    post = matrix[times >= 0]
    if len(pre) == 0 or len(post) == 0:
        return {
            "MahalanobisMagnitude": np.nan,
            "MeanAbsStandardizedChange": np.nan,
            "SignedFirstDimensionChange": np.nan,
        }
    difference = np.mean(post, axis=0) - np.mean(pre, axis=0)
    squared = float(difference @ spec.inverse_covariance @ difference)
    return {
        "MahalanobisMagnitude": float(math.sqrt(max(0.0, squared))),
        "MeanAbsStandardizedChange": float(np.mean(np.abs(difference))),
        "SignedFirstDimensionChange": float(difference[0]),
    }


def score_boundary_collection(
    boundaries: pd.DataFrame,
    specs: dict[str, RepresentationSpec],
    window_sec: int,
    rr_threshold: float,
    min_points: int,
    search_windows: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if boundaries.empty:
        return pd.DataFrame()
    search_windows = search_windows or SEARCH_WINDOWS
    rows: list[dict[str, object]] = []
    grouped_boundaries = list(boundaries.groupby("BoundaryID", sort=False))
    for boundary_number, (boundary_id, group) in enumerate(grouped_boundaries, start=1):
        if boundary_number == 1 or boundary_number % 50 == 0:
            logging.info(
                "Scoring boundary %s/%s for window=%ss RR<=%s",
                boundary_number, len(grouped_boundaries), window_sec, rr_threshold,
            )
        group = group.sort_values("RelativeCenter_sec")
        first = group.iloc[0]
        times = group["RelativeCenter_sec"].to_numpy(dtype=float)
        metadata = {
            "BoundaryID": boundary_id,
            "BoundaryKind": first["BoundaryKind"],
            "Subject": first["Subject"],
            "TransitionType": first.get("TransitionType", ""),
            "TransitionLabel": first.get("TransitionLabel", ""),
            "BoundaryOrder": int(first.get("BoundaryOrder", -1)),
            "PseudoCondition": first.get("PseudoCondition", ""),
            "CandidateID": first.get("CandidateID", ""),
            "CandidateBlockID": first.get("CandidateBlockID", ""),
            "PseudoFold": int(first.get("PseudoFold", -1)),
            "WindowLength_sec": int(window_sec),
            "RRThreshold": float(rr_threshold),
            "Observations": len(group),
            "PreObservations": int(np.sum(times < 0)),
            "PostObservations": int(np.sum(times >= 0)),
        }
        for representation, spec in specs.items():
            matrix = spec.transform(group)
            if not np.isfinite(matrix).all():
                continue
            magnitude = fixed_boundary_magnitude(times, matrix, spec)
            for search_name, (search_start, search_end) in search_windows.items():
                # The primary abrupt-change methods are repeated for every requested
                # representation. More computationally intensive gradual/trend
                # detectors are prespecified for the reduced and independently
                # derived PCA representations, which are the nonredundant primary
                # multivariate analyses.
                methods = {
                    "LegacyBIC_fixed6": legacy_bic_score(
                        times, matrix, min_points, search_start, search_end
                    ),
                    "CovIC": covariance_ic_score(
                        times, matrix, spec, min_points, search_start, search_end
                    ),
                    "BinSeg_L2": binseg_l2_score(
                        times, matrix, min_points, search_start, search_end
                    ),
                }
                if representation in {"reduced", "independent_pca"}:
                    methods.update(
                        {
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
                    )
                for method, result in methods.items():
                    rows.append(
                        {
                            **metadata,
                            "Representation": representation,
                            "Dimensions": len(spec.output_columns),
                            "SearchWindow": search_name,
                            "SearchStart_sec": search_start,
                            "SearchEnd_sec": search_end,
                            "Method": method,
                            **result,
                            "EstimatedLag1Rho": spec.lag1_rho,
                            "EffectiveSampleSize": effective_sample_size(len(group), spec.lag1_rho),
                            **magnitude,
                        }
                    )
    return pd.DataFrame(rows)


def pelt_grid_boundary_collection(
    boundaries: pd.DataFrame,
    specs: dict[str, RepresentationSpec],
    window_sec: int,
    rr_threshold: float,
    min_points: int,
    multipliers: Sequence[float],
    require_ruptures: bool,
    representations: Sequence[str] = ("reduced",),
    search_windows: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if boundaries.empty:
        return pd.DataFrame()
    search_windows = search_windows or SEARCH_WINDOWS
    rows: list[dict[str, object]] = []
    grouped_boundaries = list(boundaries.groupby("BoundaryID", sort=False))
    for boundary_number, (boundary_id, group) in enumerate(grouped_boundaries, start=1):
        if boundary_number == 1 or boundary_number % 50 == 0:
            logging.info(
                "PELT grid boundary %s/%s for window=%ss RR<=%s",
                boundary_number, len(grouped_boundaries), window_sec, rr_threshold,
            )
        group = group.sort_values("RelativeCenter_sec")
        first = group.iloc[0]
        times = group["RelativeCenter_sec"].to_numpy(dtype=float)
        metadata = {
            "BoundaryID": boundary_id,
            "BoundaryKind": first["BoundaryKind"],
            "Subject": first["Subject"],
            "TransitionType": first.get("TransitionType", ""),
            "TransitionLabel": first.get("TransitionLabel", ""),
            "BoundaryOrder": int(first.get("BoundaryOrder", -1)),
            "PseudoCondition": first.get("PseudoCondition", ""),
            "CandidateID": first.get("CandidateID", ""),
            "CandidateBlockID": first.get("CandidateBlockID", ""),
            "PseudoFold": int(first.get("PseudoFold", -1)),
            "WindowLength_sec": int(window_sec),
            "RRThreshold": float(rr_threshold),
        }
        for representation in representations:
            if representation not in specs:
                continue
            spec = specs[representation]
            matrix = spec.transform(group)
            if not np.isfinite(matrix).all():
                continue
            whitened = np.asarray(matrix, dtype=float) @ spec.whitening
            magnitude = fixed_boundary_magnitude(times, matrix, spec)
            for model in ["l2", "l1", "rbf"]:
                base_penalty = pelt_base_penalty(
                    len(whitened), whitened.shape[1], spec.lag1_rho, model
                )
                cost = SegmentCost(whitened, model)
                algorithm = None
                if rpt is not None:
                    algorithm = rpt.Pelt(
                        model=model, min_size=min_points, jump=1
                    ).fit(whitened)
                elif require_ruptures:
                    raise ImportError(
                        "ruptures is required by --require-ruptures. "
                        "Install with: pip install ruptures"
                    )
                for multiplier in multipliers:
                    penalty = float(multiplier) * base_penalty
                    if algorithm is not None:
                        predicted = algorithm.predict(pen=penalty)
                        breakpoints = [
                            int(value) for value in predicted if int(value) < len(whitened)
                        ]
                        segments = list(zip(
                            [0] + breakpoints, breakpoints + [len(whitened)]
                        ))
                        segmented_cost = float(sum(
                            cost.cost(a, b) for a, b in segments
                        ))
                        no_change_cost = float(cost.cost(0, len(whitened)))
                        implementation = "ruptures.Pelt"
                    else:
                        exact = exact_penalized_segmentation_from_cost(
                            cost, penalty, min_points
                        )
                        breakpoints = [int(value) for value in exact["Breakpoints"]]
                        segmented_cost = float(exact["SegmentedCost"])
                        no_change_cost = float(exact["NoChangeCost"])
                        implementation = str(exact["Implementation"])
                    for search_name, (search_start, search_end) in search_windows.items():
                        valid = [
                            split for split in breakpoints
                            if split < len(times)
                            and float(times[split]) >= search_start
                            and float(times[split]) <= search_end
                        ]
                        split = valid[0] if valid else -1
                        rows.append(
                            {
                                **metadata,
                                "Representation": representation,
                                "SearchWindow": search_name,
                                "SearchStart_sec": search_start,
                                "SearchEnd_sec": search_end,
                                "CostModel": model,
                                "Method": f"PELT_{model.upper()}",
                                "DetectedAtPenalty": bool(valid),
                                "SplitIndex": int(split),
                                "CandidateLatency_sec": (
                                    float(times[split]) if split >= 0 else np.nan
                                ),
                                "Score": float(no_change_cost - segmented_cost),
                                "Penalty": penalty,
                                "PenaltyMultiplier": float(multiplier),
                                "BasePenalty": base_penalty,
                                "NumBreakpoints": len(breakpoints),
                                "Breakpoints": ";".join(
                                    str(value) for value in breakpoints
                                ),
                                "NoChangeCost": no_change_cost,
                                "SegmentedCost": segmented_cost,
                                "Implementation": implementation,
                                **magnitude,
                            }
                        )
    return pd.DataFrame(rows)
