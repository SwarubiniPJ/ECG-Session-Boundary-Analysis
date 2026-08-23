# Matched pseudo-boundary calibration of ECG dynamics across affective-session transitions

This repository contains the feature-engineering, statistical-analysis, validation, and manuscript-figure code for a participant-aware analysis of ECG-derived cardiovascular dynamics around affective video-session boundaries.

## Scientific scope

The analysis is organized around three findings:

1. **Primary physiological finding:** A-to-NA session boundaries were associated with consistently larger covariance-adjusted cardiovascular change magnitude than NA-to-A boundaries.
2. **Primary methodological finding:** detection rates at real session boundaries were not consistently greater than rates at matched stable-session pseudo-boundaries, demonstrating the importance of empirical null calibration.
3. **Secondary timing finding:** participant-pooled 45-s A-to-NA analyses identified an internally supported cardiovascular departure within the first 0–45-s post-boundary interval, but the timing family did not remain supported after Holm correction across all prespecified timing tests.

The repository uses the term **session-boundary-associated cardiovascular change**. It does not claim direct anxiety detection, exact anxiety onset, or physiological recovery time.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── .gitignore
├── LICENSE_CHOICE.md
├── src/
│   ├── feature_engineering/
│   │   ├── config.example.py
│   │   ├── windowing.py
│   │   ├── signal_quality.py
│   │   ├── feature_extractor.py
│   │   ├── build_master_csv.py
│   │   └── utils.py
│   └── analysis/
│       ├── ecg_transition_analysis_timing_v4.py
│       └── ecg_transition_v4/
│           └── <copy the complete final V4 package here>
├── scripts/
│   ├── run_feature_extraction.sh
│   ├── run_timing_v4.sh
│   ├── prepare_data_from_v4_zip.py
│   ├── generate_all_figures_standardized.py
│   └── generate_methodology_pipeline.py
├── validation/
│   ├── validate_installation.py
│   ├── validate_population_timing.py
│   └── CODE_VALIDATION.md
├── environment/
│   ├── requirements-analysis.txt
│   ├── requirements-feature-extraction.txt
│   └── README.md
├── data/
│   ├── README.md
│   ├── raw/                         # not committed
│   ├── processed/                   # master CSV not committed unless archived separately
│   └── derived/manuscript/
│       ├── main/
│       ├── supplementary/
│       ├── main_tables/
│       └── supplementary_tables/
├── results/
│   ├── README.md
│   ├── analysis_manifest.json
│   ├── PRIMARY_RESULTS_SUMMARY_V4.txt
│   ├── manuscript/                  # small machine-readable result tables
│   └── full_run/                    # not committed; archive externally
├── figures/
│   ├── main/
│   └── supplementary/
├── manuscript/                      # optional during peer review
│   ├── sections/
│   ├── tables/
│   └── references/
└── docs/
    ├── REPRODUCIBILITY.md
    ├── DATA_DICTIONARY.md
    ├── FIGURE_TABLE_MANIFEST.csv
    ├── METHODS.md
    └── CHANGELOG.md
```

## Data

The source recordings are available from the public **Anxiety Dataset 2022** Figshare record:

- https://figshare.com/articles/dataset/Anxiety_Dataset_2022/19875217

Raw participant recordings are not redistributed in this repository. Download them from Figshare and place them in a local, ignored directory described in `data/README.md`.

Small de-identified derived datasets required to reproduce manuscript figures and tables may be committed under `data/derived/manuscript/`. The full master feature table and full V4 result archive should be deposited in a permanent research archive and linked from this README.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements-analysis.txt
```

Feature extraction also requires NeuroKit2 and AntroPy. Freeze their exact versions from the original feature-extraction environment before the reproducibility release:

```bash
python -m pip install -r environment/requirements-feature-extraction.txt
```

## Workflow

### 1. Build the master ECG/HRV feature table

Copy the final feature-engineering files into `src/feature_engineering/`, edit a local configuration file, and run:

```bash
python src/feature_engineering/build_master_csv.py
```

Do not commit a configuration containing personal absolute paths. Keep `config.py` local and commit only `config.example.py`.

### 2. Validate the environment and input table

```bash
python validation/validate_installation.py \
  --input "/absolute/path/to/Master_ECG_HRV_Features.csv"

python validation/validate_population_timing.py
```

### 3. Run the final analysis

```bash
bash scripts/run_timing_v4.sh \
  "/absolute/path/to/Master_ECG_HRV_Features.csv" \
  "/absolute/path/to/Nature_Timing_Validated_Results"
```

The full manuscript run evaluates 30-, 45-, and 60-s windows; RR-correction thresholds of 5%, 10%, and 20%; matched pseudo-boundaries; simulations; participant bootstrap and permutation inference; alternative feature representations; multiple change-point methods; sequence position; and end-to-end participant omission.

### 4. Prepare manuscript datasets and regenerate figures

```bash
python scripts/prepare_data_from_v4_zip.py \
  --source "/absolute/path/to/Nature_Timing_Validated_Results.zip" \
  --output-dir data/derived/manuscript

python scripts/generate_all_figures_standardized.py \
  --data-dir data/derived/manuscript \
  --output-dir figures
```

## Reproducibility records

The reported run should include:

- `analysis_manifest.json`;
- the input master-table SHA-256 checksum;
- the random seed;
- exact package versions;
- complete command-line arguments;
- `PRIMARY_RESULTS_SUMMARY_V4.txt`;
- curated manuscript CSV files;
- final figure-generation code.

See `docs/REPRODUCIBILITY.md`.

## Interpretation constraints

- Candidate times are window-centre locations, not instantaneous physiological events.
- A 45-s window centred at 22.5 s represents approximately the first 0–45 s after the nominal session boundary.
- Real-boundary findings must be interpreted relative to both stable-A and stable-NA pseudo controls.
- The fixed video order prevents separation of clip identity, sequence position, habituation, fatigue, and cumulative exposure.
- ECG alone cannot isolate anxiety from video change, attention, valence, respiration, movement, expectancy, or orienting responses.

## Citation

A `CITATION.cff` file is included. Replace the repository URL, release version, DOI, and publication DOI before making the archival release.

## Contact

- Swarubini P J, Khalifa University (swarubinipj@gmail.com)
- Mohamed Elgendi, Khalifa University (moe.elgendi@ku.ac.ae)

