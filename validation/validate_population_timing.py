#!/usr/bin/env python3
"""Synthetic recovery check for the V4 shared-timing fitter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecg_transition_v4.population_timing import SharedTimingFitter


def main() -> None:
    rng = np.random.default_rng(20260816)
    times = np.arange(-60.0, 61.0, 5.0)
    true_time = 20.0
    rows: list[dict[str, object]] = []
    for participant in range(19):
        intercept = rng.normal(0.0, 0.4, size=3)
        slope = rng.normal(0.0, 0.05, size=3)
        for time_sec in times:
            step = float(time_sec >= true_time)
            hinge = max((time_sec - true_time) / 60.0, 0.0)
            value = (
                intercept
                + slope * (time_sec / 60.0)
                + step * np.array([0.8, -0.5, 0.6])
                + hinge * np.array([0.4, 0.2, -0.3])
                + rng.normal(0.0, 0.15, size=3)
            )
            rows.append(
                {
                    "Subject": f"S{participant:02d}",
                    "RelativeCenter_sec": time_sec,
                    "V1": value[0],
                    "V2": value[1],
                    "V3": value[2],
                }
            )

    fitter = SharedTimingFitter(
        window_sec=30,
        min_unique_times_per_side=3,
        min_participants=10,
    )
    fit, _ = fitter.fit(pd.DataFrame(rows), ["V1", "V2", "V3"], 0.0, 60.0)
    if fit is None:
        raise SystemExit("Synthetic timing model was not estimable.")
    if abs(fit.candidate_time_sec - true_time) > 5.0:
        raise SystemExit(
            f"Expected recovery within 5 s of {true_time:g}; obtained "
            f"{fit.candidate_time_sec:g}."
        )
    print(
        "Population timing validation passed: "
        f"true={true_time:g}s, estimated={fit.candidate_time_sec:g}s, "
        f"score={fit.score:.6f}."
    )


if __name__ == "__main__":
    main()
