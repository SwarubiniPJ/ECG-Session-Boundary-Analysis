from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import ALL23, FEATURE_FAMILIES, REDUCED_DIRECT_CANDIDATES, RepresentationSpec
from .utils import ensure_dir, effective_sample_size


def balanced_stable_sample(
    stable: pd.DataFrame,
    columns: Sequence[str],
    seed: int,
    max_per_stratum: int = 40,
) -> pd.DataFrame:
    finite = stable[list(columns)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    source = stable.loc[finite].copy()
    if source.empty:
        raise RuntimeError("No complete stable rows are available for representation fitting.")
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for _, group in source.groupby(["Subject", "ConditionCode"], sort=True):
        take = min(len(group), int(max_per_stratum))
        if take == len(group):
            pieces.append(group)
        else:
            indices = rng.choice(group.index.to_numpy(), size=take, replace=False)
            pieces.append(group.loc[np.sort(indices)])
    return pd.concat(pieces, ignore_index=True)


def derive_reduced_feature_set(
    stable: pd.DataFrame,
    abs_threshold: float,
    output_dir: Path | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    zcols = [f"{feature}_z" for feature in REDUCED_DIRECT_CANDIDATES]
    sample = balanced_stable_sample(stable, zcols, seed=731, max_per_stratum=50)
    correlation = sample[zcols].corr(method="spearman")
    correlation.index = REDUCED_DIRECT_CANDIDATES
    correlation.columns = REDUCED_DIRECT_CANDIDATES
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for feature in REDUCED_DIRECT_CANDIDATES:
        if not selected:
            selected.append(feature)
            rows.append(
                {
                    "Feature": feature,
                    "Selected": True,
                    "RepresentativeFor": feature,
                    "MaximumAbsCorrelationWithSelected": np.nan,
                    "Reason": "first prespecified direct measure",
                }
            )
            continue
        correlations = correlation.loc[feature, selected].abs()
        representative = str(correlations.idxmax())
        maximum = float(correlations.max())
        keep = bool(maximum < abs_threshold)
        if keep:
            selected.append(feature)
        rows.append(
            {
                "Feature": feature,
                "Selected": keep,
                "RepresentativeFor": feature if keep else representative,
                "MaximumAbsCorrelationWithSelected": maximum,
                "Reason": (
                    f"abs Spearman correlation below {abs_threshold:.2f}"
                    if keep
                    else f"redundant with {representative} at abs rho={maximum:.3f}"
                ),
            }
        )
    selection = pd.DataFrame(rows)
    if output_dir is not None:
        ensure_dir(output_dir)
        correlation.to_csv(output_dir / "stable_reduced_candidate_spearman.csv")
        selection.to_csv(output_dir / "reduced_feature_selection.csv", index=False)
        (output_dir / "reduced_feature_list.json").write_text(
            json.dumps(selected, indent=2), encoding="utf-8"
        )
    return selected, correlation, selection


def covariance_whitening_matrix(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def estimate_lag1_rho_direct(stable: pd.DataFrame, columns: Sequence[str]) -> float:
    correlations: list[float] = []
    for _, group in stable.groupby(["Subject", "Session"], sort=False):
        group = group.sort_values("CenterTime_sec")
        matrix = group[list(columns)].to_numpy(dtype=float)
        if len(matrix) < 5:
            continue
        for index in range(matrix.shape[1]):
            x = matrix[:-1, index]
            y = matrix[1:, index]
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 4:
                continue
            if np.std(x[finite]) <= 1e-12 or np.std(y[finite]) <= 1e-12:
                continue
            rho = float(np.corrcoef(x[finite], y[finite])[0, 1])
            if np.isfinite(rho):
                correlations.append(rho)
    if not correlations:
        return 0.0
    return float(np.clip(np.median(correlations), 0.0, 0.98))


def estimate_lag1_rho_transformed(
    stable: pd.DataFrame,
    spec: RepresentationSpec,
) -> float:
    correlations: list[float] = []
    for _, group in stable.groupby(["Subject", "Session"], sort=False):
        group = group.sort_values("CenterTime_sec")
        if len(group) < 5:
            continue
        matrix = spec.transform(group)
        for index in range(matrix.shape[1]):
            x = matrix[:-1, index]
            y = matrix[1:, index]
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 4:
                continue
            if np.std(x[finite]) <= 1e-12 or np.std(y[finite]) <= 1e-12:
                continue
            rho = float(np.corrcoef(x[finite], y[finite])[0, 1])
            if np.isfinite(rho):
                correlations.append(rho)
    if not correlations:
        return 0.0
    return float(np.clip(np.median(correlations), 0.0, 0.98))


def _fit_covariance(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimator = LedoitWolf().fit(matrix)
    covariance = np.asarray(estimator.covariance_, dtype=float)
    inverse = np.linalg.pinv(covariance)
    whitening = covariance_whitening_matrix(covariance)
    return covariance, inverse, whitening


def fit_representation_specs(
    stable_full: pd.DataFrame,
    window_sec: int,
    step_sec: int,
    reduced_features: Sequence[str],
    pca_variance_target: float,
    seed: int,
    output_dir: Path | None = None,
) -> tuple[dict[str, RepresentationSpec], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stride = max(1, int(math.ceil(window_sec / step_sec)))
    stable_nonoverlap = stable_full[
        stable_full["WindowIndex"].astype(int) % stride == 0
    ].copy()
    if len(stable_nonoverlap) < 100:
        stable_nonoverlap = stable_full.copy()

    representation_columns: dict[str, list[str]] = {
        "all23": [f"{feature}_z" for feature in ALL23],
        "reduced": [f"{feature}_z" for feature in reduced_features],
        "heart_rate_rr": [f"{feature}_z" for feature in FEATURE_FAMILIES["heart_rate_rr"]],
        "hrv": [f"{feature}_z" for feature in FEATURE_FAMILIES["hrv"]],
        "ecg_morphology": [f"{feature}_z" for feature in FEATURE_FAMILIES["ecg_morphology"]],
    }

    specs: dict[str, RepresentationSpec] = {}
    audit_rows: list[dict[str, object]] = []
    for name, columns in representation_columns.items():
        sample = balanced_stable_sample(
            stable_nonoverlap, columns, seed=seed + window_sec + len(name), max_per_stratum=40
        )
        matrix = sample[columns].to_numpy(dtype=float)
        covariance, inverse, whitening = _fit_covariance(matrix)
        rho = estimate_lag1_rho_direct(stable_full, columns)
        specs[name] = RepresentationSpec(
            name=name,
            source_columns=list(columns),
            output_columns=list(columns),
            covariance=covariance,
            inverse_covariance=inverse,
            whitening=whitening,
            lag1_rho=rho,
            stable_rows=len(matrix),
        )
        audit_rows.append(
            {
                "WindowLength_sec": window_sec,
                "Representation": name,
                "Dimensions": len(columns),
                "StableRows": len(matrix),
                "EstimatedLag1Rho": rho,
                "TypicalNEff_n20": effective_sample_size(20, rho),
                "Likelihood": "fixed Ledoit-Wolf Gaussian covariance; multivariate window is observation unit",
                "NoChangeMeanParameters": len(columns),
                "OneChangeMeanParameters": 2 * len(columns),
                "BreakpointSearchParameter": 1,
                "ScalarFeatureCountUsedAsN": False,
            }
        )
        if output_dir is not None:
            np.savetxt(
                output_dir / f"covariance_{name}_w{window_sec}.csv",
                covariance,
                delimiter=",",
            )

    all23_columns = [f"{feature}_z" for feature in ALL23]
    pca_sample = balanced_stable_sample(
        stable_nonoverlap, all23_columns, seed=seed + window_sec * 17, max_per_stratum=40
    )
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(pca_sample[all23_columns].to_numpy(dtype=float))
    full = PCA(n_components=min(len(all23_columns), len(x_fit)), svd_solver="full")
    full.fit(x_fit)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = max(3, int(np.searchsorted(cumulative, pca_variance_target) + 1))
    n_components = min(n_components, full.n_components_)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(x_fit)
    transformed_fit = pca.transform(x_fit)
    covariance, inverse, whitening = _fit_covariance(transformed_fit)
    output_columns = [f"StablePC{index + 1}" for index in range(n_components)]
    pca_spec = RepresentationSpec(
        name="independent_pca",
        source_columns=all23_columns,
        output_columns=output_columns,
        covariance=covariance,
        inverse_covariance=inverse,
        whitening=whitening,
        lag1_rho=0.0,
        stable_rows=len(transformed_fit),
        scaler=scaler,
        pca=pca,
    )
    pca_spec.lag1_rho = estimate_lag1_rho_transformed(stable_full, pca_spec)
    specs["independent_pca"] = pca_spec
    audit_rows.append(
        {
            "WindowLength_sec": window_sec,
            "Representation": "independent_pca",
            "Dimensions": n_components,
            "StableRows": len(transformed_fit),
            "EstimatedLag1Rho": pca_spec.lag1_rho,
            "TypicalNEff_n20": effective_sample_size(20, pca_spec.lag1_rho),
            "Likelihood": "stable-session PCA followed by fixed Ledoit-Wolf Gaussian covariance",
            "NoChangeMeanParameters": n_components,
            "OneChangeMeanParameters": 2 * n_components,
            "BreakpointSearchParameter": 1,
            "ScalarFeatureCountUsedAsN": False,
        }
    )

    variance = pd.DataFrame(
        {
            "WindowLength_sec": window_sec,
            "Component": output_columns,
            "ExplainedVarianceRatio": pca.explained_variance_ratio_,
            "CumulativeExplainedVariance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    loadings = pd.DataFrame(
        pca.components_.T, index=ALL23, columns=output_columns
    ).reset_index(names="Feature")
    loadings.insert(0, "WindowLength_sec", window_sec)
    scaler_table = pd.DataFrame(
        {
            "WindowLength_sec": window_sec,
            "Feature": ALL23,
            "StablePCA_Mean": scaler.mean_,
            "StablePCA_Scale": scaler.scale_,
        }
    )
    audit = pd.DataFrame(audit_rows)

    if output_dir is not None:
        ensure_dir(output_dir)
        variance.to_csv(output_dir / f"pca_variance_w{window_sec}.csv", index=False)
        loadings.to_csv(output_dir / f"pca_loadings_w{window_sec}.csv", index=False)
        scaler_table.to_csv(output_dir / f"pca_scaler_w{window_sec}.csv", index=False)
        audit.to_csv(output_dir / f"representation_statistical_audit_w{window_sec}.csv", index=False)
    return specs, variance, loadings, audit
