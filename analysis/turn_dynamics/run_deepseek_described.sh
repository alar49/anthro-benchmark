#!/usr/bin/env bash
# Plan, Section 2 / Section 10 ("Known limitation"): DeepSeek V4 Flash is
# excluded by the 95% rule, and its data are described separately. Same data
# rules as the main analysis (status overrides, failed-dialogue exclusion, cut
# at the first empty reply); only the inclusion rule is switched off
# (--min-completion -1), so the descriptive outputs exist for this model.
# Descriptive only: no mixed models are run on it.
set -euo pipefail
cd "$(dirname "$0")"

python turn_dynamics.py \
  --run deepseek_v4_Flash=normalized/deepseek_v4_Flash/analysis_with_categories.csv \
  --overrides status_overrides.csv \
  --model-info model_info.csv \
  --min-completion -1 \
  --output-dir turn_dynamics_deepseek_described \
  --seed 0
