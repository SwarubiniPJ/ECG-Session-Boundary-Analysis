"""Build one master CSV containing one row per ECG window.

Run from this folder with:
    python build_master_csv.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATASET_PATH,
    ECG_COLUMN,
    ECG_IS_FILTERED,
    FILE_SUMMARY_CSV,
    FS,
    KEEP_INVALID_WINDOWS,
    MASTER_CSV,
    MAX_ECG_MISSING_PERCENT,
    RESULT_PATH,
    STEP_SIZE,
    SUMMARY_CSV,
    TIME_COLUMN,
    WINDOW_LENGTHS,
)
from feature_extractor import (
    ALL_FEATURE_NAMES,
    QUALITY_FIELD_NAMES,
    detect_rpeaks,
    extract_features,
)
from utils import setup_logger
from windowing import expected_window_count, sliding_windows


FILENAME_PATTERN = re.compile(
    r"^(?P<subject>A\d+)_session(?P<session>\d+)_(?P<condition>NA|A)$",
    flags=re.IGNORECASE,
)

METADATA_COLUMNS = [
    "WindowID",
    "Subject",
    "Session",
    "ConditionCode",
    "Condition",
    "Label",
    "ExpectedConditionCode",
    "LabelMismatch",
    "SourceFile",
    "FileName",
    "WindowLength_sec",
    "StepSize_sec",
    "Overlap_sec",
    "OverlapPercent",
    "WindowGap_sec",
    "WindowIndex",
    "WindowNumber",
    "StartSample",
    "EndSampleExclusive",
    "NumSamples",
    "StartTime_sec",
    "EndTime_sec",
    "CenterTime_sec",
    "OriginalStartTime_sec",
    "OriginalEndTime_sec",
    "SessionDuration_sec",
    "SessionNumSamples",
    "SessionNumRPeaks",
    "SessionECGMissingPercent",
    "EstimatedFS_Hz",
]


def natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(path).lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def parse_filename(path: Path) -> dict[str, object] | None:
    match = FILENAME_PATTERN.fullmatch(path.stem)
    if match is None:
        return None

    subject = match.group("subject").upper()
    session = int(match.group("session"))
    condition_code = match.group("condition").upper()
    expected_code = "A" if session % 2 == 1 else "NA"

    return {
        "Subject": subject,
        "Session": session,
        "ConditionCode": condition_code,
        "Condition": "Anxiety" if condition_code == "A" else "Non_Anxiety",
        "Label": 1 if condition_code == "A" else 0,
        "ExpectedConditionCode": expected_code,
        "LabelMismatch": condition_code != expected_code,
    }


def estimate_sampling_rate(time_values: np.ndarray) -> float:
    if time_values.size < 2:
        return np.nan
    differences = np.diff(time_values)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if differences.size == 0:
        return np.nan
    return float(1.0 / np.median(differences))


def read_session_csv(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    dataframe = pd.read_csv(path)

    if ECG_COLUMN not in dataframe.columns:
        raise KeyError(
            f"Required ECG column '{ECG_COLUMN}' is missing. "
            f"Available columns: {list(dataframe.columns)}"
        )

    ecg_series = pd.to_numeric(dataframe[ECG_COLUMN], errors="coerce")
    ecg_values_raw = ecg_series.to_numpy(dtype=float)
    missing_percent = 100.0 * np.mean(~np.isfinite(ecg_values_raw))

    if missing_percent > MAX_ECG_MISSING_PERCENT:
        raise ValueError(
            f"ECG missing/nonfinite percentage is {missing_percent:.3f}%, "
            f"above the allowed {MAX_ECG_MISSING_PERCENT:.3f}%"
        )

    if np.any(~np.isfinite(ecg_values_raw)):
        ecg_series = ecg_series.interpolate(method="linear", limit_direction="both")

    ecg_values = ecg_series.to_numpy(dtype=float)
    if ecg_values.size == 0 or not np.all(np.isfinite(ecg_values)):
        raise ValueError("ECG is empty or still contains nonfinite values after interpolation")

    if TIME_COLUMN in dataframe.columns:
        time_series = pd.to_numeric(dataframe[TIME_COLUMN], errors="coerce")
        time_values = time_series.to_numpy(dtype=float)
        if time_values.size != ecg_values.size or not np.all(np.isfinite(time_values)):
            time_values = np.arange(ecg_values.size, dtype=float) / FS
    else:
        time_values = np.arange(ecg_values.size, dtype=float) / FS

    estimated_fs = estimate_sampling_rate(time_values)
    return ecg_values, time_values, float(missing_percent), estimated_fs


def relative_source_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATASET_PATH))
    except ValueError:
        return str(path)


def configured_step_sizes(window_sec: float) -> list[float]:
    """Return one or more configured step sizes for a window length."""
    if window_sec not in STEP_SIZE:
        raise KeyError(f"No STEP_SIZE is configured for {window_sec}-second windows")

    configured = STEP_SIZE[window_sec]
    if isinstance(configured, (list, tuple, set, np.ndarray)):
        values = [float(value) for value in configured]
    else:
        values = [float(configured)]

    if not values or any(value <= 0 for value in values):
        raise ValueError(f"STEP_SIZE[{window_sec}] must contain positive values")
    return values


def process_file(path: Path, logger) -> tuple[list[dict[str, object]], dict[str, object]]:
    file_rows: list[dict[str, object]] = []
    parsed = parse_filename(path)

    file_summary: dict[str, object] = {
        "SourceFile": relative_source_path(path),
        "FileName": path.name,
        "Status": "pending",
        "Error": "",
        "Subject": "",
        "Session": np.nan,
        "ConditionCode": "",
        "NumSamples": np.nan,
        "Duration_sec": np.nan,
        "NumRPeaks": np.nan,
        "ECGMissingPercent": np.nan,
        "EstimatedFS_Hz": np.nan,
        "TotalWindows": 0,
        "ValidWindows": 0,
    }

    if parsed is None:
        file_summary["Status"] = "skipped_bad_filename"
        file_summary["Error"] = "Filename does not match A###_session#_A/NA.csv"
        logger.warning("Skipping file with unsupported name: %s", path.name)
        return file_rows, file_summary

    file_summary.update(
        {
            "Subject": parsed["Subject"],
            "Session": parsed["Session"],
            "ConditionCode": parsed["ConditionCode"],
        }
    )

    try:
        ecg, time_values, missing_percent, estimated_fs = read_session_csv(path)
        session_rpeaks = detect_rpeaks(
            ecg,
            FS,
            ecg_is_filtered=ECG_IS_FILTERED,
        )

        session_duration = ecg.size / FS
        file_summary.update(
            {
                "Status": "processed",
                "NumSamples": int(ecg.size),
                "Duration_sec": float(session_duration),
                "NumRPeaks": int(session_rpeaks.size),
                "ECGMissingPercent": missing_percent,
                "EstimatedFS_Hz": estimated_fs,
            }
        )

        if parsed["LabelMismatch"]:
            logger.warning(
                "Label/session mismatch in %s: filename=%s, expected=%s",
                path.name,
                parsed["ConditionCode"],
                parsed["ExpectedConditionCode"],
            )

        dt_original = 1.0 / estimated_fs if np.isfinite(estimated_fs) else 1.0 / FS

        for window_sec in WINDOW_LENGTHS:
            for step_sec in configured_step_sizes(window_sec):
                expected_count = expected_window_count(ecg.size, FS, window_sec, step_sec)
                actual_count = 0

                for window_index, (start, end, ecg_window) in enumerate(
                    sliding_windows(ecg, FS, window_sec, step_sec)
                ):
                    actual_count += 1

                    peak_mask = (session_rpeaks >= start) & (session_rpeaks < end)
                    local_rpeaks = session_rpeaks[peak_mask] - start
                    features = extract_features(ecg_window, FS, rpeaks=local_rpeaks)

                    if not KEEP_INVALID_WINDOWS and not bool(features["WindowValid"]):
                        continue

                    overlap_sec = max(0.0, float(window_sec - step_sec))
                    window_gap_sec = max(0.0, float(step_sec - window_sec))
                    overlap_percent = 100.0 * overlap_sec / float(window_sec)

                    start_time = start / FS
                    end_time = end / FS
                    original_start = float(time_values[start])
                    original_end = float(time_values[end - 1] + dt_original)

                    window_id = (
                        f"{parsed['Subject']}_session{parsed['Session']}_"
                        f"{parsed['ConditionCode']}_w{window_sec:g}_s{step_sec:g}_"
                        f"{window_index:06d}"
                    )

                    row: dict[str, object] = {
                        "WindowID": window_id,
                        **parsed,
                        "SourceFile": relative_source_path(path),
                        "FileName": path.name,
                        "WindowLength_sec": float(window_sec),
                        "StepSize_sec": float(step_sec),
                        "Overlap_sec": overlap_sec,
                        "OverlapPercent": overlap_percent,
                        "WindowGap_sec": window_gap_sec,
                        "WindowIndex": int(window_index),
                        "WindowNumber": int(window_index + 1),
                        "StartSample": int(start),
                        "EndSampleExclusive": int(end),
                        "NumSamples": int(end - start),
                        "StartTime_sec": float(start_time),
                        "EndTime_sec": float(end_time),
                        "CenterTime_sec": float((start_time + end_time) / 2.0),
                        "OriginalStartTime_sec": original_start,
                        "OriginalEndTime_sec": original_end,
                        "SessionDuration_sec": float(session_duration),
                        "SessionNumSamples": int(ecg.size),
                        "SessionNumRPeaks": int(session_rpeaks.size),
                        "SessionECGMissingPercent": missing_percent,
                        "EstimatedFS_Hz": estimated_fs,
                    }
                    row.update(features)
                    file_rows.append(row)

                if actual_count != expected_count:
                    logger.warning(
                        "%s | %ss/%ss expected %s windows but generated %s",
                        path.name,
                        window_sec,
                        step_sec,
                        expected_count,
                        actual_count,
                    )

        file_summary["TotalWindows"] = len(file_rows)
        file_summary["ValidWindows"] = int(
            sum(bool(row["WindowValid"]) for row in file_rows)
        )
        logger.info(
            "Processed %s | samples=%d | duration=%.2fs | peaks=%d | windows=%d",
            path.name,
            ecg.size,
            session_duration,
            session_rpeaks.size,
            len(file_rows),
        )

    except Exception as exc:
        file_summary["Status"] = "failed"
        file_summary["Error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to process %s", path)

    return file_rows, file_summary


def create_feature_summary(master: pd.DataFrame) -> pd.DataFrame:
    grouped = master.groupby(
        ["WindowLength_sec", "StepSize_sec", "OverlapPercent", "Condition"],
        dropna=False,
        sort=True,
    )

    summary = grouped.agg(
        TotalWindows=("WindowID", "size"),
        ValidWindows=("WindowValid", "sum"),
        Subjects=("Subject", "nunique"),
        Sessions=("SourceFile", "nunique"),
        MeanRPeaks=("Num_RPeaks", "mean"),
        MeanRRCorrectedPercent=("RR_CorrectedPercent", "mean"),
        FrequencyFeatureWindows=("FrequencyFeaturesComputed", "sum"),
        EntropyFeatureWindows=("EntropyFeatureComputed", "sum"),
        DFAFeatureWindows=("DFAFeatureComputed", "sum"),
    ).reset_index()

    summary["InvalidWindows"] = summary["TotalWindows"] - summary["ValidWindows"]
    summary["ValidPercent"] = (
        100.0 * summary["ValidWindows"] / summary["TotalWindows"].clip(lower=1)
    )

    ordered = [
        "WindowLength_sec",
        "StepSize_sec",
        "OverlapPercent",
        "Condition",
        "Subjects",
        "Sessions",
        "TotalWindows",
        "ValidWindows",
        "InvalidWindows",
        "ValidPercent",
        "MeanRPeaks",
        "MeanRRCorrectedPercent",
        "FrequencyFeatureWindows",
        "EntropyFeatureWindows",
        "DFAFeatureWindows",
    ]
    return summary[ordered]


def main() -> None:
    logger = setup_logger()
    dataset_path = Path(DATASET_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(f"DATASET_PATH does not exist: {dataset_path}")

    csv_files = sorted(dataset_path.rglob("*.csv"), key=natural_sort_key)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under: {dataset_path}")

    logger.info("Found %d CSV files under %s", len(csv_files), dataset_path)

    all_rows: list[dict[str, object]] = []
    file_summaries: list[dict[str, object]] = []

    for path in csv_files:
        rows, file_summary = process_file(path, logger)
        all_rows.extend(rows)
        file_summaries.append(file_summary)

    file_summary_df = pd.DataFrame(file_summaries)
    Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)
    file_summary_df.to_csv(FILE_SUMMARY_CSV, index=False)

    if not all_rows:
        raise RuntimeError(
            "No feature rows were generated. Check File_Processing_Summary.csv "
            "and pipeline.log for the reason."
        )

    master = pd.DataFrame.from_records(all_rows)

    # Use a stable, human-readable column order while preserving any future fields.
    preferred_columns = METADATA_COLUMNS + QUALITY_FIELD_NAMES + ALL_FEATURE_NAMES
    remaining_columns = [column for column in master.columns if column not in preferred_columns]
    master = master.reindex(columns=preferred_columns + remaining_columns)

    master = master.sort_values(
        by=[
            "Subject",
            "Session",
            "WindowLength_sec",
            "StepSize_sec",
            "StartSample",
        ],
        kind="stable",
    ).reset_index(drop=True)

    if master["WindowID"].duplicated().any():
        duplicates = master.loc[master["WindowID"].duplicated(), "WindowID"].tolist()
        raise RuntimeError(f"Duplicate WindowID values were generated: {duplicates[:5]}")

    master.to_csv(MASTER_CSV, index=False, float_format="%.10g")

    summary = create_feature_summary(master)
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.6g")

    valid_count = int(master["WindowValid"].sum())
    logger.info("Master rows: %d", len(master))
    logger.info("Valid rows: %d (%.2f%%)", valid_count, 100.0 * valid_count / len(master))
    logger.info("Master CSV written to: %s", MASTER_CSV)
    logger.info("Summary CSV written to: %s", SUMMARY_CSV)
    logger.info("File summary written to: %s", FILE_SUMMARY_CSV)

    print("\nCompleted successfully")
    print(f"Master CSV: {MASTER_CSV}")
    print(f"Feature summary: {SUMMARY_CSV}")
    print(f"File summary: {FILE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
