#!/usr/bin/env bash
# Exploratory rerun (plan, Section 10, added after the results): the Step 6
# analysis without each dialogue's final rated turn, which is usually a
# closing exchange. Same --run lines, overrides, model info and seed as
# run_turn_dynamics.sh; the inputs are copies of normalized/ without those rows.
# Then: Rscript turn_dynamics_models.R turn_dynamics_drop_final_turn
set -euo pipefail
cd "$(dirname "$0")"

RUNS=(
  --run deepseek_v4_Flash=normalized/deepseek_v4_Flash/analysis_with_categories.csv
  --run gpt_oss_120b=normalized/gpt_oss_120b/analysis_with_categories.csv
  --run llama3.3_70b=normalized/llama3.3_70b/analysis_with_categories.csv
  --run gemma4_31b=normalized/gemma4_31b/analysis_with_categories.csv
  --run qwen3.8_27b=normalized/qwen3.8_27b/analysis_with_categories.csv
  --run mistral_nemo=normalized/mistral_nemo/analysis_with_categories.csv
  --run qwen3.5_9b=normalized/qwen3.5_9b/analysis_with_categories.csv
  --run llama3.1_8b=normalized/llama3.1_8b/analysis_with_categories.csv
  --run gemma4_e2b=normalized/gemma4_e2b/analysis_with_categories.csv
  --run phi4_mini_3.8=normalized/phi4_mini_3.8/analysis_with_categories.csv
  --run gemini3.7_flash=normalized/gemini3.7_flash/analysis_with_categories.csv
  --run mistral_small_4_seed42_only=normalized/mistral_small_4_seed42_only/analysis_with_categories.csv
)
DROPPED=("${RUNS[@]/=normalized\//=normalized_drop_final/}")

python drop_final_turn.py "${RUNS[@]}" --out-dir normalized_drop_final

python turn_dynamics.py "${DROPPED[@]}" --overrides status_overrides.csv \
  --model-info model_info.csv --output-dir turn_dynamics_drop_final_turn --seed 0
