#!/usr/bin/env bash
# Step 10 of "How to run": the two Mistral Small 4 checks (plan Section 11).
# --scenario-col prompt_text: the only column with the same value for the same
# scenario in all three runs (dialogue_index is 0-95 in seed42_only and test
# but 0-959 in all960). In all960 one prompt appears three times (dialogue_index
# 505, 506, 508; 506 is in seed42), so check 1's rest has 862 dialogues, not 864.
set -euo pipefail
cd "$(dirname "$0")"

python subsample_checks.py \
  --full "mistral_small_4_all960_full=normalized/STEP 10 ONLY/mistral_small_4_all960_full/analysis_with_categories.csv" \
  --subsample mistral_small_4_seed42_only=normalized/mistral_small_4_seed42_only/analysis_with_categories.csv \
  --test "mistral_small_4_test=normalized/STEP 10 ONLY/mistral_small_4_test/analysis_with_categories.csv" \
  --scenario-col prompt_text \
  --overrides status_overrides.csv \
  --output-dir subsample_checks_results --seed 0
