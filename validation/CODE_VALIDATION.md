# V4 Code Validation Notes

These checks validate the V4 implementation and output structure. They are not
manuscript results and do not replace a complete run using the replicate counts
specified in `README.md`.

## V4-specific checks completed

- All Python modules pass `compileall`.
- The command-line parser exposes all population-timing controls.
- A synthetic 19-participant, three-dimensional shared change injected at 20 s
  was recovered at 20 s by `SharedTimingFitter`.
- A no-change synthetic trajectory remained estimable but produced a much
  smaller score; the code does not label such a candidate significant without
  pseudo calibration.
- The population timing module was exercised on the attached master ECG/HRV
  table using 30-s windows, RR correction <=20%, the reduced representation,
  matched stable-A and stable-NA pseudo blocks, participant bootstrap, timing
  LOPO and timing simulation.
- That real-data smoke test wrote timing summary, score profile, pseudo-null,
  bootstrap, direction-comparison and LOPO part files for both NA-to-A and
  A-to-NA.
- The reporting functions were tested with timing tables and generated the V4
  timing worksheets and timing figures.
- The complete package launcher and every module compile successfully.

## Interpretation of the real-data smoke test

The low-replicate smoke test returned a candidate centre for both directions.
It deliberately kept validation separate: an unsupported candidate remained
labelled `descriptive_candidate_not_validated`. Smoke-test numerical values
must not be used in the manuscript.

## What was not completed in this execution environment

- `ruptures` was not installed, so the literal `ruptures.Pelt` path was not run
  here. Install `requirements_timing_v4.txt` and use `--require-ruptures` for the
  manuscript analysis.
- A complete manuscript-scale V4 run was not allowed to finish in this
  environment because the retained V3 detector suite, PELT grids, simulations,
  end-to-end LOPO and the new timing null/bootstrap analyses are computationally
  extensive. Run `run_timing_v4.sh` or the command in `README.md` on the study
  workstation.
- The code cannot guarantee a statistically validated NA-to-A time. It improves
  sensitivity by pooling participants, but the final status still depends on
  whether the real timing score exceeds both matched pseudo-control nulls.

## Study-design limitations not removable by code

The dataset contains 19 healthy participants, a fixed clip order, no external
validation cohort, and no concurrent time-resolved subjective anxiety,
respiration or movement reference suitable for separating anxiety from other
boundary-associated responses.
