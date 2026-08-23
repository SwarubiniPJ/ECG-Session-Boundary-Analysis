from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALL23, ALLOWED_WINDOWS, NEUTRAL_DIRECTION_LABELS, REQUIRED_COLUMNS
from .utils import parse_bool, safe_numeric


def read_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Master CSV not found: {path}")
    data = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise KeyError(f"Master CSV is missing required columns: {sorted(missing)}")
    missing_features = [feature for feature in ALL23 if feature not in data.columns]
    if missing_features:
        raise KeyError(f"Master CSV is missing analysis features: {missing_features}")

    data["Subject"] = data["Subject"].astype(str)
    data["Session"] = pd.to_numeric(data["Session"], errors="raise").astype(int)
    data["ConditionCode"] = data["ConditionCode"].astype(str).str.upper()
    condition_text = data["Condition"].astype(str).str.lower()
    bad = data["ConditionCode"].isin(["", "NAN", "NONE"])
    data.loc[bad, "ConditionCode"] = np.where(
        condition_text.loc[bad].str.contains("non"), "NA", "A"
    )
    data["WindowLength_sec"] = pd.to_numeric(
        data["WindowLength_sec"], errors="raise"
    ).astype(int)
    data["StepSize_sec"] = pd.to_numeric(data["StepSize_sec"], errors="raise").astype(int)
    data["WindowIndex"] = pd.to_numeric(data["WindowIndex"], errors="raise").astype(int)
    data["WindowValid"] = parse_bool(data["WindowValid"])
    for column in [
        "StartTime_sec", "EndTime_sec", "CenterTime_sec", "SessionDuration_sec",
        "RR_CorrectedPercent",
    ]:
        data[column] = safe_numeric(data[column])
    for feature in ALL23:
        data[feature] = safe_numeric(data[feature])

    available = set(data["WindowLength_sec"].dropna().astype(int).unique())
    absent = set(ALLOWED_WINDOWS).difference(available)
    if absent:
        raise ValueError(f"Master CSV does not contain required windows: {sorted(absent)}")
    return data


def session_information(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Subject", "Session", "ConditionCode", "Condition", "SessionDuration_sec"
    ]
    return (
        data[columns]
        .drop_duplicates(["Subject", "Session"])
        .sort_values(["Subject", "Session"])
        .reset_index(drop=True)
    )


def build_real_boundary_inventory(data: pd.DataFrame) -> pd.DataFrame:
    sessions = session_information(data)
    rows: list[dict[str, object]] = []
    for subject, group in sessions.groupby("Subject", sort=True):
        lookup = {int(row.Session): row for row in group.itertuples(index=False)}
        numbers = sorted(lookup)
        for pre_session, post_session in zip(numbers[:-1], numbers[1:]):
            if post_session != pre_session + 1:
                continue
            pre = lookup[pre_session]
            post = lookup[post_session]
            pre_code = str(pre.ConditionCode).upper()
            post_code = str(post.ConditionCode).upper()
            direction = f"{pre_code}_to_{post_code}"
            boundary_id = f"{subject}_B{pre_session}_{direction}"
            rows.append(
                {
                    "BoundaryID": boundary_id,
                    "Subject": str(subject),
                    "PreSession": int(pre_session),
                    "PostSession": int(post_session),
                    "BoundaryOrder": int(pre_session),
                    "TransitionType": direction,
                    "TransitionLabel": NEUTRAL_DIRECTION_LABELS.get(direction, direction),
                    "PreConditionCode": pre_code,
                    "PostConditionCode": post_code,
                    "PreDuration_sec": float(pre.SessionDuration_sec),
                    "PostDuration_sec": float(post.SessionDuration_sec),
                }
            )
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise RuntimeError("No adjacent-session boundaries were found.")
    return inventory


def configuration_audit(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["WindowLength_sec", "StepSize_sec", "ConditionCode"], as_index=False)
        .agg(
            Rows=("WindowID", "size"),
            ValidWindows=("WindowValid", "sum"),
            Participants=("Subject", "nunique"),
            Sessions=("Session", "nunique"),
            MeanRRCorrectedPercent=("RR_CorrectedPercent", "mean"),
            MedianRRCorrectedPercent=("RR_CorrectedPercent", "median"),
        )
    )
