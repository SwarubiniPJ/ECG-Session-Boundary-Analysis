#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/Master_ECG_HRV_Features.csv [output_folder]" >&2
  exit 2
fi

INPUT=$1
OUTPUT=${2:-Nature_Timing_Validated_Results_V4}

python ecg_transition_analysis_timing_v4.py \
  --input "$INPUT" \
  --output-root "$OUTPUT" \
  --windows 30 45 60 \
  --step 5 \
  --rr-thresholds 5 10 20 \
  --pseudo-controls-per-boundary 50 \
  --pseudo-folds 4 \
  --bootstrap 5000 \
  --permutations 10000 \
  --null-simulations 400 \
  --power-simulations 100 \
  --timing-windows 30 45 60 \
  --timing-rr-thresholds 20 \
  --timing-representations reduced independent_pca \
  --timing-endpoints departure_magnitude signed_trajectory \
  --timing-search-windows post_only anticipatory \
  --timing-pseudo-draws 5000 \
  --timing-bootstrap 5000 \
  --timing-power-simulations 100 \
  --timing-simulation-effect-sizes 0.5 1.0 1.5 \
  --timing-simulation-affected-fraction 0.5 \
  --timing-primary-window 30 \
  --timing-primary-rr-threshold 20 \
  --timing-primary-representation reduced \
  --timing-primary-endpoint departure_magnitude \
  --lopo-rr-thresholds 20 \
  --require-ruptures
