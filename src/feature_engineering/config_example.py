"""Configuration for ECG windowing and feature extraction."""

from pathlib import Path

# -----------------------------------------------------------------------------
# Dataset and output paths
# -----------------------------------------------------------------------------

DATASET_PATH = Path("/path/to/ECG_Filtered_Sessions")
RESULT_PATH = Path("/path/to/Results_Feature_Engineering")

MASTER_CSV = RESULT_PATH / "Master_ECG_HRV_Features.csv"
SUMMARY_CSV = RESULT_PATH / "Feature_Extraction_Summary.csv"
FILE_SUMMARY_CSV = RESULT_PATH / "File_Processing_Summary.csv"
LOG_FILE = RESULT_PATH / "pipeline.log"

# -----------------------------------------------------------------------------
# Input CSV structure
# -----------------------------------------------------------------------------
ECG_COLUMN = "ECG_filtered"
TIME_COLUMN = "Time_seconds"
FS = 500
ECG_IS_FILTERED = True

# Expected filename example: A101_session1_A.csv or A101_session2_NA.csv
# Odd sessions are expected to be anxiety; even sessions are non-anxiety.

# -----------------------------------------------------------------------------
# Window configurations
# -----------------------------------------------------------------------------
# A STEP_SIZE value can be either one number or a list/tuple of numbers.
# Example for testing two overlaps for a 10-second window: 10: [2, 5]
WINDOW_LENGTHS = [5, 10, 15, 20, 30, 45, 60]
STEP_SIZE = {
    5: 1,
    10: 2,
    15: 3,
    20: 5,
    30: 5,
    45: 5,
    60: 5,
}

# -----------------------------------------------------------------------------
# R-peak detection
# -----------------------------------------------------------------------------
RPEAK_METHOD = "neurokit"
CORRECT_RPEAK_ARTIFACTS = True

# -----------------------------------------------------------------------------
# ECG and RR quality control
# -----------------------------------------------------------------------------
MAX_ECG_MISSING_PERCENT = 1.0
RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0
HAMPEL_HALF_WINDOW = 5
HAMPEL_N_SIGMA = 3.0
MAX_RR_CORRECTED_PERCENT = 20.0

# At least 3 RR intervals (4 R-peaks) are required for a window to be considered
# minimally usable. Feature-specific functions impose stricter requirements.
MIN_RR_INTERVALS_FOR_VALID_WINDOW = 3

# -----------------------------------------------------------------------------
# Feature-specific minimum data requirements
# -----------------------------------------------------------------------------
# Frequency-domain values from 60-second windows should be treated as
# exploratory. Classical short-term HRV spectral analysis normally uses longer,
# stationary recordings. Set this to 300 for a strict 5-minute policy.
MIN_WINDOW_SEC_FOR_FREQUENCY = 60.0
MIN_RR_INTERVALS_FOR_FREQUENCY = 30

MIN_RR_INTERVALS_FOR_SAMPLE_ENTROPY = 20

# DFA alpha1 is defined over short scales, commonly 4-16 beats. To estimate all
# those scales with enough segments, considerably more RR intervals are needed.
MIN_RR_INTERVALS_FOR_DFA_ALPHA1 = 160

# Keep every complete window in the master table. Invalid windows receive NaN
# feature values plus WindowValid=False and a reason, rather than disappearing.
KEEP_INVALID_WINDOWS = True
