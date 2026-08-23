from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import LOG1P_FEATURES


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(output_root: Path) -> None:
    ensure_dir(output_root)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_root / "timing_v4_analysis.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    for noisy in ["matplotlib", "fontTools", "PIL", "openpyxl", "numba"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def package_versions() -> dict[str, str]:
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
        "matplotlib", "openpyxl", "ruptures",
    ]
    out = {"python": platform.python_version()}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "not installed"
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(text: str, seed: int = 0) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    return mapped.fillna(False).astype(bool)


def robust_center_scale(values: Iterable[float]) -> tuple[float, float, str]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, "unavailable"
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = 1.4826 * mad
    method = "MAD"
    if not np.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.quantile(x, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
        method = "IQR"
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
        method = "SD"
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
        method = "constant_fallback"
    return center, scale, method


def transform_feature_values(values: pd.Series | np.ndarray, feature: str) -> np.ndarray:
    x = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    x[~np.isfinite(x)] = np.nan
    if feature in LOG1P_FEATURES:
        x[x < 0] = np.nan
        x = np.log1p(x)
    return x


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    p = safe_numeric(values)
    result = pd.Series(np.nan, index=p.index, dtype=float)
    finite = p.notna()
    if not finite.any():
        return result
    x = p.loc[finite].to_numpy(dtype=float)
    order = np.argsort(x)
    ranked = x[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result.loc[finite] = restored
    return result


def quantile_higher(values: Sequence[float], probability: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    try:
        return float(np.quantile(x, probability, method="higher"))
    except TypeError:
        return float(np.quantile(x, probability, interpolation="higher"))


def effective_sample_size(n: int, rho: float) -> float:
    if n <= 1:
        return float(max(1, n))
    clipped = float(np.clip(rho, 0.0, 0.98))
    estimate = n * (1.0 - clipped) / (1.0 + clipped)
    return float(np.clip(estimate, 2.0, float(n)))


def normal_ci(estimate: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return estimate - z * se, estimate + z * se


def participant_cluster_bootstrap(
    data: pd.DataFrame,
    statistic,
    n_bootstrap: int,
    seed: int,
    participant_col: str = "Subject",
) -> tuple[float, float, float]:
    clean = data.dropna(subset=[participant_col]).copy()
    if clean.empty:
        return np.nan, np.nan, np.nan
    subjects = np.asarray(sorted(clean[participant_col].astype(str).unique()), dtype=object)
    estimate = float(statistic(clean))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    groups = {s: clean[clean[participant_col].astype(str).eq(str(s))] for s in subjects}
    for index in range(n_bootstrap):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for copy_index, subject in enumerate(sampled):
            part = groups[subject].copy()
            part[participant_col] = f"{subject}__boot{copy_index}"
            parts.append(part)
        boot[index] = float(statistic(pd.concat(parts, ignore_index=True)))
    low, high = np.nanquantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)


def participant_sign_flip(
    participant_differences: Sequence[float],
    permutations: int,
    seed: int,
) -> tuple[float, float, int]:
    differences = np.asarray(participant_differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return np.nan, np.nan, 0
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(differences)))
    null = np.mean(signs * differences[None, :], axis=1)
    pvalue = float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1))
    return observed, pvalue, int(len(differences))


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def combine_gzip_csv_parts(parts: Sequence[Path], output_path: Path) -> None:
    """Combine gzip-compressed CSV parts without loading them into memory."""
    import gzip
    import shutil

    valid = [Path(part) for part in parts if Path(part).exists() and Path(part).stat().st_size > 0]
    if not valid:
        return
    ensure_dir(output_path.parent)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as destination:
        wrote_header = False
        for part in valid:
            with gzip.open(part, "rt", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header:
                    continue
                if not wrote_header:
                    destination.write(header)
                    wrote_header = True
                shutil.copyfileobj(source, destination, length=1024 * 1024)


def combine_plain_csv_parts(parts: Sequence[Path], output_path: Path) -> None:
    """Combine plain-text CSV parts without loading them into memory."""
    import shutil

    valid = [Path(part) for part in parts if Path(part).exists() and Path(part).stat().st_size > 0]
    if not valid:
        return
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        wrote_header = False
        for part in valid:
            with part.open("r", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header:
                    continue
                if not wrote_header:
                    destination.write(header)
                    wrote_header = True
                shutil.copyfileobj(source, destination, length=1024 * 1024)
