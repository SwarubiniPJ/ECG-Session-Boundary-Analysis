#!/usr/bin/env python3
"""Validate V4 dependencies and the master CSV schema before a full run."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd

from ecg_transition_v4.config import ALLOWED_WINDOWS, REQUIRED_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--allow-pelt-fallback",
        action="store_true",
        help="Allow the exact dynamic-programming PELT-objective fallback when ruptures is unavailable.",
    )
    args = parser.parse_args()

    required_modules = [
        "numpy", "pandas", "scipy", "sklearn", "statsmodels",
        "matplotlib", "openpyxl", "ruptures",
    ]
    failures: list[str] = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"OK dependency: {module}")
        except Exception as exc:
            if module == "ruptures" and args.allow_pelt_fallback:
                print(f"OPTIONAL dependency unavailable: {module}: {exc}")
                print("The exact dynamic-programming PELT-objective fallback will be used.")
            else:
                failures.append(f"{module}: {exc}")
                print(f"MISSING dependency: {module}: {exc}")

    path = args.input.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    missing = sorted(REQUIRED_COLUMNS.difference(header.columns))
    if missing:
        failures.append(f"missing master columns: {missing}")
    else:
        print("OK master CSV required columns")

    configurations = pd.read_csv(
        path, usecols=["WindowLength_sec", "StepSize_sec"], low_memory=False
    ).drop_duplicates()
    for window in ALLOWED_WINDOWS:
        present = bool(
            ((configurations["WindowLength_sec"] == window)
             & (configurations["StepSize_sec"] == 5)).any()
        )
        print(f"{'OK' if present else 'MISSING'} configuration: {window}s/5s")
        if not present:
            failures.append(f"missing {window}s/5s configuration")

    if failures:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(failures))
    print("Validation passed. The manuscript run can be started.")


if __name__ == "__main__":
    main()
