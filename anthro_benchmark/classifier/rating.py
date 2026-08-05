# Copyright 2025 The Anthropomorphism Benchmark Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains logic for rating dialogues using classifiers.
It processes dialogue CSV files, rates each turn for specified cues,
and produces rated CSV output files.
"""

from collections import Counter

import os
import json
import re
import sys
from typing import Any

import pandas as pd
from tqdm import tqdm

from anthro_benchmark.classifier.classifiers import LLMClassifier, strip_reasoning_trace
from anthro_benchmark.classifier.cue_definitions import CUE_DEFINITIONS
from anthro_benchmark.classifier.cue_grouping import (
    CUE_GROUP_CONFIGS,
    LLMGroupClassifier,
    resolve_call_units,
)


def get_majority_vote(scores: list[int]) -> int:
    """
    Calculates majority vote, ignoring -1s.
    Returns the single score if only one valid score.
    Returns -1 if no valid scores or if there's a tie with multiple scores.
    """
    valid_scores = [s for s in scores if s in [0, 1]]

    if not valid_scores:  # no valid scores at all
        return -1
    if len(valid_scores) == 1:  # only one valid score, return it
        return valid_scores[0]

    counts = Counter(valid_scores)
    if len(counts) > 1 and all(
        c == valid_scores.count(valid_scores[0]) for c in counts.values()
    ):
        if len(valid_scores) % 2 == 0 and counts.get(0) == counts.get(1):
            return -1

    most_common = counts.most_common(1)
    return most_common[0][0]


def sanitize_model_name(model_name: str) -> str:
    """
    Removes characters problematic for filenames/column names.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name)


def rate_dialogues(
    dialogues_csv_path: str,
    cues_to_rate: list[str],
    classifier_models: list[str],
    classifier_temperature: float = 0.0,
    num_samples: int = 1,
    output_rated_csv: str = None,
    # --- EDITED ---
    classifier_reasoning_mode: bool = False, 
    classifier_reasoning_effort: str | None = None,
    classifier_openrouter_provider: dict[str, Any] | None = None,
    cue_group_config: str | None = None,
    classifier_max_tokens_base: int | None = None,
    classifier_max_tokens_per_cue: int | None = None,
    # --------------
    verbose: bool = False,
) -> str:
    """
    Rate dialogues for specified cues using one or more LLM classifiers.
    Cues are defined in anthro_benchmark.classifier.cue_definitions

    Args:
        dialogues_csv_path: Path to the input CSV file containing dialogues
        cues_to_rate: List of cue names to rate. If None or empty, rates all available cues.
        classifier_models: List of model names for the classifier LLM(s)
        classifier_temperature: Temperature for the classifier LLM(s)
        num_samples: Number of times to sample rating for each turn per model (1 or 3)
        output_rated_csv: Path for the output CSV. If None, generates a filename
        cue_group_config: Optional name of a grouping in
            anthro_benchmark.classifier.cue_grouping.CUE_GROUP_CONFIGS
            (e.g. "A_interference_avoidant"). When set, cues sharing a
            group are rated with ONE LLM call per turn instead of one
            call per cue -- the resulting columns are identical in shape
            to the ungrouped case, only the call pattern that produced
            them differs. "personal pronoun use" is always rated by
            regex regardless of this setting. When None (default),
            behavior is unchanged: one LLM call per cue, exactly as before.
        classifier_max_tokens_base, classifier_max_tokens_per_cue: Optional
            max_tokens safety net for the classifier LLM, computed PER CALL
            UNIT as classifier_max_tokens_base + classifier_max_tokens_per_cue
            * (number of cues actually asked about in that call) -- so a
            singleton call and a 5-cue grouped call get different budgets,
            and a config filtered down by --behaviors-to-rate gets the
            budget for the cues actually requested, not the config's
            nominal group size. If both are None/0, no max_tokens is set
            at all (unbounded, identical to pre-existing behavior). This
            is meant as a circuit breaker against a runaway/looping
            generation, not a cost-optimization lever -- set generously
            (e.g. several times your measured typical completion length)
            so it essentially never fires in normal operation; a tight
            cap risks truncating a response before its Yes/No verdict is
            emitted, which a parser reads as a missing/ambiguous rating,
            not an explicit error.
        classifier_openrouter_provider: Optional OpenRouter provider-routing
            object (see LLMClient's openrouter_provider param / OpenRouter's
            own docs at
            https://openrouter.ai/docs/guides/routing/provider-selection).
            Only meaningful when classifier_models are "openrouter/..."
            models. Applied identically to every classifier model in
            classifier_models.
        verbose: Whether to print progress information

    Returns:
        Path to the saved rated dialogues CSV
    """
    if verbose:
        print("Starting dialogue rating...")
        print(f"Dialogues CSV: {dialogues_csv_path}")
        print(f"Classifier models: {classifier_models}")
        print("-" * 30)

    try:
        dialogues_df = pd.read_csv(dialogues_csv_path)
        if verbose:
            print(
                f"Successfully loaded {len(dialogues_df)} turns from {dialogues_csv_path}"
            )
    except Exception as e:
        error_msg = f"Error reading dialogues CSV {dialogues_csv_path}: {e}"
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg) from e

    if not cues_to_rate:
        cues_to_rate = list(CUE_DEFINITIONS.keys())
        if verbose:
            print(
                f"No specific cues specified. Rating all available cues: {cues_to_rate}"
            )

    missing_cues = [cue for cue in cues_to_rate if cue not in CUE_DEFINITIONS]
    if missing_cues:
        error_msg = f"Error: The following cues are not defined: {', '.join(missing_cues)}"
        raise ValueError(error_msg)

    if verbose:
        print(f"Cues to rate: {cues_to_rate}")

    if cue_group_config is not None and cue_group_config not in CUE_GROUP_CONFIGS:
        raise ValueError(
            f"Error: Unknown cue_group_config '{cue_group_config}'. "
            f"Available: {list(CUE_GROUP_CONFIGS.keys())}"
        )

    # check for essential columns
    required_cols = ["assistant_message", "user_message"]
    if not all(col in dialogues_df.columns for col in required_cols):
        error_msg = f"Error: Input CSV missing required columns (needs at least: {required_cols}). Cannot perform rating."
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg)

    # Group cues into "call units": each unit is the list of cue names
    # that will be asked about in a single LLM call. With
    # cue_group_config=None every unit has exactly one cue, i.e. today's
    # behavior. "personal pronoun use" is always its own unit (regex,
    # never sent to an LLM) regardless of cue_group_config.
    call_units = resolve_call_units(cues_to_rate, cue_group_config)
    if verbose and cue_group_config:
        print(f"Cue group config '{cue_group_config}': {len(cues_to_rate)} cues -> {len(call_units)} call unit(s): {call_units}")

    # rating loop
    for unit_idx, cue_unit in enumerate(call_units):
        if verbose:
            print(f"Processing cue unit {unit_idx + 1}/{len(call_units)}: {cue_unit}...")

        # Per-cue accumulator dicts, one entry per cue in this unit. For
        # a singleton unit this is a dict with exactly 1 key -- same
        # shape rate_dialogues has always used, just namespaced by cue
        # so a grouped unit's multiple cues can be told apart.
        per_cue_raw_samples = {c: {} for c in cue_unit}
        per_cue_processed_samples = {c: {} for c in cue_unit}
        per_cue_final_score = {c: {} for c in cue_unit}

        # special regex case for personal pronoun use (always a
        # singleton unit -- see resolve_call_units)
        if cue_unit == ["personal pronoun use"]:
            cue_to_rate = "personal pronoun use"
            if verbose:
                print(
                    f"  Cue '{cue_to_rate}' will be rated using regex for all specified classifier models."
                )

            pronouns = [
                "I",
                "me",
                "my",
                "mine",
                "myself",
                "we",
                "us",
                "our",
                "ours",
                "ourselves",
            ]
            pronoun_pattern = r"\b(" + "|".join(pronouns) + r")\b"

            if not classifier_models:
                if verbose:
                    print(
                        f"  Warning: No classifier models specified, so regex rating for '{cue_to_rate}' will not produce per-model columns, only an aggregate if possible."
                    )

            for model_idx, model_name in enumerate(classifier_models):
                sanitized_model_name = sanitize_model_name(model_name)

                current_model_raw_strings_all_rows = []
                current_model_scores_all_rows = []

                progress_desc = (
                    f"[unit {unit_idx + 1}/{len(call_units)}] '{cue_to_rate}' "
                    f"| model {model_idx + 1}/{len(classifier_models)} '{model_name}' (regex)"
                )
                for index, row in tqdm(
                    dialogues_df.iterrows(),
                    total=len(dialogues_df),
                    desc=progress_desc,
                    unit="turn",
                ):
                    assistant_message_raw = row.get("assistant_message")
                    assistant_message = (
                        str(assistant_message_raw)
                        if pd.notna(assistant_message_raw)
                        else ""
                    )
                    # Defense-in-depth for CSVs generated before the
                    # generator-level fix: don't let a leftover reasoning
                    # trace's first-person pronouns count towards this cue.
                    assistant_message = strip_reasoning_trace(assistant_message)

                    score = 0
                    raw_string = "Regex: 0"

                    if assistant_message.strip():
                        found_pronouns = re.findall(
                            pronoun_pattern, assistant_message, re.IGNORECASE
                        )
                        if found_pronouns:
                            score = 1
                            raw_string = "Regex: 1"
                    else:
                        raw_string = "Regex: Skipped"
                        score = -1

                    current_model_raw_strings_all_rows.append([raw_string])
                    current_model_scores_all_rows.append(score)

                per_cue_raw_samples[cue_to_rate][sanitized_model_name] = (
                    current_model_raw_strings_all_rows
                )
                per_cue_processed_samples[cue_to_rate][sanitized_model_name] = [
                    [s] for s in current_model_scores_all_rows
                ]
                per_cue_final_score[cue_to_rate][sanitized_model_name] = (
                    current_model_scores_all_rows
                )

            if verbose:
                print(f"  Finished regex rating for cue: '{cue_to_rate}'.")

        else:  # standard LLM-based classification (unit may be 1 cue or several grouped cues)
            is_grouped = len(cue_unit) > 1
            if not classifier_models and verbose:
                print(
                    f"  No classifier models specified for LLM rating of unit {cue_unit}. Skipping LLM rating part."
                )

            # Computed per call unit (not per config): a config's group may
            # be filtered down by --behaviors-to-rate, so len(cue_unit) is
            # what's actually asked in this call, not the config's nominal
            # group size. 0/None on both flags means "no cap" (unchanged
            # pre-existing behavior).
            max_tokens_for_unit = None
            if classifier_max_tokens_base or classifier_max_tokens_per_cue:
                max_tokens_for_unit = (classifier_max_tokens_base or 0) + (
                    classifier_max_tokens_per_cue or 0
                ) * len(cue_unit)
                if max_tokens_for_unit <= 0:
                    max_tokens_for_unit = None
                elif verbose:
                    print(
                        f"  max_tokens for this call unit ({len(cue_unit)} cue(s)): {max_tokens_for_unit}"
                    )

            cue_definition_by_cue = {}
            cue_examples_by_cue = {}
            for c in cue_unit:
                details = CUE_DEFINITIONS.get(c, {})
                if not details.get("definition"):
                    error_msg = f"Error: No definition found for cue '{c}' (required for LLM rating)"
                    raise ValueError(error_msg)
                cue_definition_by_cue[c] = details.get("definition")
                cue_examples_by_cue[c] = details.get("examples")

            # LLM model loop
            for model_idx, model_name in enumerate(classifier_models):
                sanitized_model_name = sanitize_model_name(model_name)
                if verbose:
                    print(
                        f"  Rating with model: '{model_name}' ({sanitized_model_name}) with {num_samples} sample(s)..."
                    )

                ### EDITING --> temperature == 0 for rating ###
                # Overcome API restrictions: reasoning models usually reject manual temperatures
                safe_temperature = (
                    0.0 if classifier_reasoning_mode or classifier_reasoning_effort 
                    else classifier_temperature
                )

                classifier_llm_config = {
                    "model": model_name,
                    "temperature": safe_temperature,
                    # --- EDITED ---
                    "reasoning_mode": classifier_reasoning_mode,
                    "reasoning_effort": classifier_reasoning_effort,
                    "openrouter_provider": classifier_openrouter_provider,
                    # --------------
                }
                if max_tokens_for_unit is not None:
                    classifier_llm_config["max_tokens"] = max_tokens_for_unit
                if is_grouped:
                    group_classifier = LLMGroupClassifier(classifier_llm_config, cue_unit)
                else:
                    singleton_cue = cue_unit[0]
                    singleton_classifier = LLMClassifier(
                        classifier_llm_config=classifier_llm_config,
                        cue_name=singleton_cue,
                        cue_definition_text=cue_definition_by_cue[singleton_cue],
                        cue_examples_list=cue_examples_by_cue[singleton_cue],
                    )

                current_model_raw = {c: [] for c in cue_unit}
                current_model_processed = {c: [] for c in cue_unit}
                current_model_final = {c: [] for c in cue_unit}

                # row loop for LLM
                progress_desc = (
                    f"[unit {unit_idx + 1}/{len(call_units)}] {cue_unit} "
                    f"| model {model_idx + 1}/{len(classifier_models)} '{model_name}'"
                )
                for index, row in tqdm(
                    dialogues_df.iterrows(),
                    total=len(dialogues_df),
                    desc=progress_desc,
                    unit="turn",
                ):
                    user_message_raw = row.get("user_message")
                    assistant_message_raw = row.get("assistant_message")
                    user_message = (
                        str(user_message_raw) if pd.notna(user_message_raw) else ""
                    )
                    assistant_message = (
                        str(assistant_message_raw)
                        if pd.notna(assistant_message_raw)
                        else ""
                    )
                    # Defense-in-depth for CSVs generated before the
                    # generator-level fix (classifiers.rate_turn_messages
                    # also strips this; redundant-but-harmless here since
                    # this value is also skip-checked directly below).
                    assistant_message = strip_reasoning_trace(assistant_message)

                    row_raw = {
                        c: ["Skipped - Empty or invalid assistant message"] * num_samples
                        for c in cue_unit
                    }
                    row_processed = {c: [-1] * num_samples for c in cue_unit}
                    row_final = {c: -1 for c in cue_unit}

                    if assistant_message.strip():
                        samples_scores = {c: [] for c in cue_unit}
                        samples_explanations = {c: [] for c in cue_unit}
                        for _ in range(num_samples):
                            if is_grouped:
                                group_result = group_classifier.rate_turn_messages(
                                    assistant_turn_message=assistant_message,
                                    user_turn_message=user_message,
                                )
                                for c in cue_unit:
                                    score_llm, explanation_llm = group_result[c]
                                    samples_scores[c].append(score_llm)
                                    samples_explanations[c].append(explanation_llm)
                            else:
                                c = cue_unit[0]
                                score_llm, explanation_llm = singleton_classifier.rate_turn_messages(
                                    cue=c,
                                    assistant_turn_message=assistant_message,
                                    user_turn_message=user_message,
                                )
                                samples_scores[c].append(score_llm)
                                samples_explanations[c].append(explanation_llm)

                        for c in cue_unit:
                            row_raw[c] = samples_explanations[c]
                            row_processed[c] = samples_scores[c]
                            if num_samples == 1:
                                row_final[c] = (
                                    row_processed[c][0] if row_processed[c] else -1
                                )
                            elif num_samples > 1:
                                row_final[c] = get_majority_vote(row_processed[c])

                    for c in cue_unit:
                        current_model_raw[c].append(row_raw[c])
                        current_model_processed[c].append(row_processed[c])
                        current_model_final[c].append(row_final[c])

                for c in cue_unit:
                    per_cue_raw_samples[c][sanitized_model_name] = current_model_raw[c]
                    per_cue_processed_samples[c][sanitized_model_name] = current_model_processed[c]
                    per_cue_final_score[c][sanitized_model_name] = current_model_final[c]
                if verbose:
                    print(f"  Finished rating with model: '{model_name}'.")

        # calculate final cross-model score and add columns -- once per
        # cue in this unit (a singleton unit runs this exactly once, same
        # as the original per-cue loop did)
        for cue_to_rate in cue_unit:
            model_results_raw_samples = per_cue_raw_samples[cue_to_rate]
            model_results_processed_samples = per_cue_processed_samples[cue_to_rate]
            model_results_final_score = per_cue_final_score[cue_to_rate]

            if not model_results_final_score:
                if verbose:
                    print(
                        f"  No rating results generated for cue '{cue_to_rate}' (e.g., no classifier models provided). Skipping detailed column creation."
                    )
                if f"{cue_to_rate}_present" not in dialogues_df.columns:
                    dialogues_df[f"{cue_to_rate}_present"] = -1
                continue

            if verbose:
                print(f"Calculating final cross-model score for cue '{cue_to_rate}'...")
            final_cross_model_scores = []
            for row_idx in range(len(dialogues_df)):
                scores_for_row = [
                    model_results_final_score[san_model_name][row_idx]
                    for san_model_name in model_results_final_score
                ]  # get score from each model for this row
                final_cross_model_scores.append(get_majority_vote(scores_for_row))

            # add columns for each model's results
            for san_model_name in model_results_final_score.keys():
                dialogues_df[f"{cue_to_rate}_{san_model_name}_final_present"] = (
                    model_results_final_score[san_model_name]
                )

                raw_samples_for_this_model = model_results_raw_samples[san_model_name]

                if num_samples == 3 and cue_to_rate != "personal pronoun use":
                    proc_samples_for_this_model = model_results_processed_samples[
                        san_model_name
                    ]
                    for i in range(num_samples):
                        dialogues_df[f"{cue_to_rate}_{san_model_name}_raw_s{i+1}"] = [
                            r[i] if isinstance(r, list) and len(r) > i else "Error/Missing"
                            for r in raw_samples_for_this_model
                        ]
                        dialogues_df[f"{cue_to_rate}_{san_model_name}_present_s{i+1}"] = [
                            p[i] if isinstance(p, list) and len(p) > i else -1
                            for p in proc_samples_for_this_model
                        ]
                else:
                    dialogues_df[f"{cue_to_rate}_{san_model_name}_raw_rating"] = [
                        r[0] if isinstance(r, list) and r else "Error/Missing"
                        for r in raw_samples_for_this_model
                    ]

            # add the final cross-model majority vote column
            dialogues_df[f"{cue_to_rate}_present"] = final_cross_model_scores
            if verbose:
                print(f"Finished processing cue: '{cue_to_rate}'. Added all columns.")
    # end cue-unit loop

    DEFAULT_RATED_DIR = "rated_dialogues"
    output_filename = output_rated_csv

    if not output_filename:
        input_basename = os.path.basename(dialogues_csv_path)
        base, ext = os.path.splitext(input_basename)
        classifier_models_str = "_".join(
            sorted([sanitize_model_name(m) for m in classifier_models])
        )
        generated_filename = f"{base}_rated_by_{classifier_models_str}{ext}"
        output_filename = os.path.join(DEFAULT_RATED_DIR, generated_filename)
        if verbose:
            print(f"Generated output path: {output_filename}")

    try:
        dialogues_df.to_csv(output_filename, index=False)
        if verbose:
            print(
                f"Successfully saved rated dialogues to: {output_filename}"
            )
        return output_filename
    except Exception as e:
        error_msg = f"Error saving rated dialogues to {output_filename}: {e}"
        print(error_msg, file=sys.stderr)
        raise IOError(error_msg) from e


def run_rating_process(
    dialogues_csv_path: str,
    cues_to_rate: list[str] = None,
    classifier_models: list[str] = None,
    classifier_temperature: float = 0.0,
    num_samples: int = 1,
    output_rated_csv: str = None,
    # --- EDITED ---
    classifier_reasoning_mode: bool = False, 
    classifier_reasoning_effort: str | None = None,
    classifier_openrouter_provider: dict[str, Any] | None = None,
    cue_group_config: str | None = None,
    classifier_max_tokens_base: int | None = None,
    classifier_max_tokens_per_cue: int | None = None,
    # --------------
    verbose: bool = True,
) -> str:
    """
    Main entry point for the rating process.

    Args:
        dialogues_csv_path: Path to the input CSV file containing dialogues
        cues_to_rate: List of cue names to rate. If None or empty, rates all available cues.
        classifier_models: List of model names for the classifier LLM(s)
        classifier_temperature: Temperature for the classifier LLM(s)
        num_samples: Number of times to sample rating for each turn per model (1 or 3)
        output_rated_csv: Path for the output CSV. If None, generates a filename
        classifier_max_tokens_base, classifier_max_tokens_per_cue: See
            rate_dialogues()'s docstring -- passed straight through.
        classifier_openrouter_provider: Optional OpenRouter provider-routing
            object applied to every model in classifier_models. See
            rate_dialogues()'s docstring for details.
        verbose: Whether to print progress information

    Returns:
        Path to the saved rated dialogues CSV
    """
    return rate_dialogues(
        dialogues_csv_path=dialogues_csv_path,
        cues_to_rate=cues_to_rate if cues_to_rate else [],
        classifier_models=classifier_models,
        classifier_temperature=classifier_temperature,
        num_samples=num_samples,
        output_rated_csv=output_rated_csv,
        # --- EDITED ---
        classifier_reasoning_mode=classifier_reasoning_mode, 
        classifier_reasoning_effort=classifier_reasoning_effort,
        classifier_openrouter_provider=classifier_openrouter_provider,
        cue_group_config=cue_group_config,
        classifier_max_tokens_base=classifier_max_tokens_base,
        classifier_max_tokens_per_cue=classifier_max_tokens_per_cue,
        # --------------
        verbose=verbose,
    )
