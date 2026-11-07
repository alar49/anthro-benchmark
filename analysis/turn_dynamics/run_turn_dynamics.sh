#!/usr/bin/env bash
# Step 6 of "How to run": Python part of the turn analysis, one --run per model.
# Inputs are the standard copies in normalized/ (Step 2: format normalized,
# Excel encoding damage repaired, seed42_only dialogue_index mapped to 0-95);
# the original files in final_csv/ are untouched.
set -euo pipefail
cd "$(dirname "$0")"

python turn_dynamics.py \
  --run deepseek_v4_Flash=normalized/deepseek_v4_Flash/analysis_with_categories.csv \
  --run gpt_oss_120b=normalized/gpt_oss_120b/analysis_with_categories.csv \
  --run llama3.3_70b=normalized/llama3.3_70b/analysis_with_categories.csv \
  --run gemma4_31b=normalized/gemma4_31b/analysis_with_categories.csv \
  --run qwen3.8_27b=normalized/qwen3.8_27b/analysis_with_categories.csv \
  --run mistral_nemo=normalized/mistral_nemo/analysis_with_categories.csv \
  --run qwen3.5_9b=normalized/qwen3.5_9b/analysis_with_categories.csv \
  --run llama3.1_8b=normalized/llama3.1_8b/analysis_with_categories.csv \
  --run gemma4_e2b=normalized/gemma4_e2b/analysis_with_categories.csv \
  --run phi4_mini_3.8=normalized/phi4_mini_3.8/analysis_with_categories.csv \
  --run gemini3.7_flash=normalized/gemini3.7_flash/analysis_with_categories.csv \
  --run mistral_small_4_seed42_only=normalized/mistral_small_4_seed42_only/analysis_with_categories.csv \
  --overrides status_overrides.csv \
  --model-info model_info.csv \
  --output-dir turn_dynamics_results \
  --seed 0
