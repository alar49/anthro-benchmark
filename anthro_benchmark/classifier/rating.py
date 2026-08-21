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

import asyncio
import os
import json
import re
import sys
from typing import Any, Dict, Union

import pandas as pd
from tqdm import tqdm

from anthro_benchmark.classifier.classifiers import LLMClassifier, strip_reasoning_trace
from anthro_benchmark.classifier.cue_definitions import CUE_DEFINITIONS
from anthro_benchmark.classifier.cue_grouping import (
    CUE_GROUP_CONFIGS,
    LLMGroupClassifier,
    resolve_call_units,
)
from anthro_benchmark.core.llm_client import BudgetGuard, BudgetExceededError, RateLimiter
from anthro_benchmark.core.io_utils import atomic_to_csv


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


async def _rate_rows_concurrently_async(row_fn, rows, max_concurrency, desc):
    semaphore = asyncio.Semaphore(max_concurrency)
    progress_bar = tqdm(total=len(rows), desc=desc, unit="turn")

    async def _run_one(row):
        async with semaphore:
            try:
                result = await asyncio.to_thread(row_fn, row)
            except BudgetExceededError as e:
                # Converted to a value, not re-raised: this lets the
                # whole batch finish (everything else in flight or
                # already queued still gets its result recorded) instead
                # of asyncio.gather aborting on the first failure and
                # losing every other row's work. Any OTHER exception is
                # deliberately NOT caught here -- it propagates out of
                # gather() the same way an uncaught exception in the
                # original sequential loop would have crashed the whole
                # rate_dialogues() call, just discovered at batch
                # granularity instead of per-row.
                result = e
        progress_bar.update(1)
        return result

    try:
        return await asyncio.gather(*(_run_one(row) for row in rows))
    finally:
        progress_bar.close()


def _rate_rows_concurrently(row_items, row_fn, max_concurrency, desc):
    """row_items: list of (index, row) pairs, e.g. from
    dialogues_df.iterrows(). row_fn(row) -> (row_raw, row_processed,
    row_final); may raise BudgetExceededError. Returns a list, one entry
    per row IN THE SAME ORDER as row_items, where each entry is either
    row_fn's normal return value or a BudgetExceededError instance --
    order must be preserved because callers index results positionally
    to line up with dialogues_df's rows."""
    rows = [row for _, row in row_items]
    return asyncio.run(
        _rate_rows_concurrently_async(row_fn, rows, max_concurrency, desc)
    )


async def _rate_rows_chunked_async(
    row_fn, rows, max_concurrency, desc, on_chunk_complete=None
):
    """
    Like _rate_rows_concurrently_async, but dispatches in chunks of
    max_concurrency, fully awaiting each chunk (asyncio.gather) before
    the next chunk starts -- trading some throughput (a straggler in one
    chunk blocks the next chunk from starting even if other concurrency
    slots are free, same tradeoff as generator.py's strict_batch_ordering)
    for a guaranteed-complete, gap-free prefix of results at every chunk
    boundary.

    That boundary is what makes safe incremental checkpointing possible.
    Continuous dispatch (_rate_rows_concurrently_async, via
    asyncio.gather over ALL rows at once) has no such boundary: every
    row is in flight simultaneously and results only become available
    all together when the whole batch finishes, so there's no
    well-defined "everything so far" to checkpoint mid-batch.

    on_chunk_complete(results_so_far), if given, is called synchronously
    after each chunk resolves and BEFORE the next chunk is dispatched,
    with the full growing list of results accumulated so far (in row
    order) -- callers use this to checkpoint-save.
    """
    progress_bar = tqdm(total=len(rows), desc=desc, unit="turn")
    results = []
    try:
        for chunk_start in range(0, len(rows), max_concurrency):
            chunk = rows[chunk_start : chunk_start + max_concurrency]

            async def _run_one(row):
                try:
                    result = await asyncio.to_thread(row_fn, row)
                except BudgetExceededError as e:
                    # Same rationale as _rate_rows_concurrently_async:
                    # converted to a value so the rest of this chunk
                    # still finishes and gets recorded.
                    result = e
                progress_bar.update(1)
                return result

            chunk_results = await asyncio.gather(*(_run_one(row) for row in chunk))
            results.extend(chunk_results)
            if on_chunk_complete is not None:
                on_chunk_complete(results)
    finally:
        progress_bar.close()
    return results


def _rate_rows_chunked(
    row_items, row_fn, max_concurrency, desc, on_chunk_complete=None
):
    """Strict-batch-ordering counterpart to _rate_rows_concurrently --
    see _rate_rows_chunked_async's docstring. Same (index, row) input
    shape and same "results in row_items order" output guarantee."""
    rows = [row for _, row in row_items]
    return asyncio.run(
        _rate_rows_chunked_async(
            row_fn, rows, max_concurrency, desc, on_chunk_complete
        )
    )


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
    classifier_timeout: float | None = None,
    classifier_max_retries: int = 5,
    classifier_max_timeout_retries: int | None = None,
    classifier_initial_backoff: float = 2.0,
    classifier_max_backoff: float = 60.0,
    rate_limiter: Union[RateLimiter, Dict[str, RateLimiter], None] = None,
    budget_guard: BudgetGuard | None = None,
    max_concurrency: int = 1,
    strict_batch_ordering: bool = False,
    incremental_save: bool = False,
    resume: bool = False,
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
        classifier_timeout: Optional per-call ceiling in seconds, applied
            identically to every classifier model in classifier_models.
            None (default) preserves prior behavior exactly (no
            client-side cap). See LLMClient's timeout param in
            llm_client.py for exactly what happens when it's exceeded,
            and how that interacts with retries.
        classifier_max_retries: Maximum retry attempts per classifier call
            for transient errors OTHER than a timeout (429, 5xx,
            connection drops). Applied identically to every model in
            classifier_models. Default 5 matches LLMClient's own default,
            so omitting this changes nothing versus prior behavior (this
            simply wasn't configurable here before).
        classifier_max_timeout_retries: Maximum retry attempts
            specifically for a classifier_timeout expiry, independent of
            classifier_max_retries -- see LLMClient's max_timeout_retries
            param in llm_client.py for the full rationale (in short: a
            timeout retry has an unknown, possibly-nonzero provider-side
            cost never visible to budget_guard's dollar tracking, unlike
            a 429/connection error, so it's often worth capping tighter).
            None (default) uses the same value as classifier_max_retries.
        classifier_initial_backoff, classifier_max_backoff: Backoff
            bounds (seconds) for classifier call retries. Defaults match
            LLMClient's own prior hardcoded values, so omitting these
            changes nothing versus prior behavior.
        rate_limiter: Optional RateLimiter (see llm_client.py), for
            staying under a provider's own requests/tokens-per-minute
            limits instead of finding out you exceeded them via a 429 or
            a hung connection. Either:
              - a single RateLimiter, shared across every classifier
                model and cue unit in this run (use when every model in
                classifier_models draws on the same provider quota, e.g.
                one OpenRouter account/key for all of them); or
              - a {model_name: RateLimiter} dict, giving each classifier
                model its own independent limiter (use when models are
                mixed across providers/tiers with genuinely different
                limits -- e.g. a free-tier model alongside a paid one,
                where throttling them together would needlessly slow the
                paid model down to the free one's pace). A model name
                present in classifier_models but missing from this dict
                gets no rate limiting.
        classifier_openrouter_provider: Optional OpenRouter provider-routing
            object (see LLMClient's openrouter_provider param / OpenRouter's
            own docs at
            https://openrouter.ai/docs/guides/routing/provider-selection).
            Only meaningful when classifier_models are "openrouter/..."
            models. Applied identically to every classifier model in
            classifier_models.
        budget_guard: Optional BudgetGuard shared across every cue unit and
            every classifier model in this run (one guard total, same
            pattern as generate sharing one guard between its user and
            target LLMs). When the cap is hit, rating stops after finishing
            whatever cue unit was in progress (its columns are still
            written using whichever models/rows completed) rather than
            continuing to attempt calls that would immediately fail --
            already-rated rows and already-completed cue units are not
            lost.
        max_concurrency: How many rows to rate at once, per (cue unit,
            classifier model) combination. Default 1 reproduces the
            original fully-sequential behavior exactly. Values > 1 run
            rows concurrently via asyncio.to_thread -- ratings have no
            cross-row dependency (unlike generate's dialogue turns), so
            this is where concurrency helps most. Real OS-thread
            concurrency under the hood, same as generate's
            max_concurrency -- see BudgetGuard's docstring in
            llm_client.py for the overshoot caveat when combined with a
            budget cap.
        strict_batch_ordering: Only relevant when max_concurrency > 1.
            Default False dispatches all rows in a (cue unit, model)
            combination continuously (a new row starts the instant a
            concurrency slot frees) for maximum throughput. True
            dispatches in chunks of max_concurrency, one chunk fully
            finished before the next starts -- at a measured throughput
            cost (same order of magnitude as generator.py's
            strict_batch_ordering: roughly 1-15% under ordinary latency
            variance, more when a straggler is in a chunk), but this is
            what incremental_save needs to checkpoint safely under
            concurrency -- see _rate_rows_chunked_async's docstring for
            why continuous dispatch can't. Automatically enabled if
            incremental_save is set together with max_concurrency > 1.
        incremental_save: Default False preserves prior behavior exactly
            (the CSV is written once, after the entire run finishes -- a
            crash/hang/kill before that point saves nothing at all, so a
            run that dies partway through a long cue-unit/multi-model
            session loses everything). When True, the CSV is rewritten:
              - after each row completes, in sequential mode
                (max_concurrency <= 1);
              - after each completed chunk of rows within a (cue unit,
                model)'s row loop, under concurrency (forces
                strict_batch_ordering=True -- see above);
              - after each cue unit's columns are fully written (this
                already happened at this point regardless of
                incremental_save; it's just also checkpointed here).
            The finer, within-a-model checkpoints write that model's
            PARTIAL per-row results into its own
            "{cue}_{model}_final_present" column as they resolve -- the
            cross-model aggregate "{cue}_present" column is only
            computed once ALL models finish for that cue (unchanged from
            before), so an interrupted run's checkpoint shows real
            progress per-model even for a cue unit that didn't finish,
            while the "official" verdict for that cue remains whatever
            it was before this unit started (-1/absent if never rated).
            Each save is atomic (temp file + rename -- see
            io_utils.atomic_to_csv), so an interruption during the save
            itself can't corrupt the previous good checkpoint either.
        resume: Default False. Requires incremental_save=True (raises
            ValueError otherwise). If output_rated_csv (or the generated
            default path) already exists, merges in its per-model
            "{cue}_{model}_final_present" columns -- matched by
            (dialogue_id, turn_pair_index), which are stable here since
            rating.py doesn't sample its own row set, unlike generate.py
            -- and skips re-rating any (row, cue, model) combination
            that already has a VALID (0 or 1) value there, reusing the
            existing value instead. A combination whose prior value is
            -1 (couldn't parse a valid rating from the response -- see
            calculate_summary_stats' handling of this same sentinel in
            analysis.py) is deliberately NOT treated as done: it's
            RE-ATTEMPTED, since -1 means the previous attempt failed to
            produce a usable verdict, not that it succeeded with a
            negative-looking answer. A cue+model's prior result is
            reused regardless of which --cue-group-config unit it was
            originally rated under; "{cue}_{model}_final_present" means
            the same thing either way. Reused rows' raw-explanation/
            per-sample columns are replaced with a placeholder noting
            they were reused (only the final per-model score is
            checkpointed incrementally, not the per-sample detail behind
            it) -- the final score itself is exactly the original value,
            never approximated. If the output path doesn't exist yet,
            resume has no effect (a normal fresh run). Raises ValueError
            if the existing file has no per-model "*_final_present"
            columns (predates resume support, wasn't produced with
            incremental_save, or wasn't produced by this pipeline).
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

    # Resolved here (not at the very end, as before this refactor)
    # because incremental_save needs a stable path to checkpoint to
    # throughout the run, not just after it finishes.
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

    if incremental_save and max_concurrency > 1 and not strict_batch_ordering:
        # Same rationale as generator.py's DialogueGenerator.__init__ --
        # see its comment for the full explanation. Continuous dispatch
        # has no point during a batch where "everything so far" is a
        # well-defined, safely-checkpointable set; strict_batch_ordering
        # (chunk-by-chunk, each fully awaited) does.
        print(
            "incremental_save=True with max_concurrency > 1 requires "
            "strict_batch_ordering for safe checkpointing -- enabling "
            "it automatically. This trades some throughput for "
            "guaranteed-safe incremental saves; see "
            "_rate_rows_chunked_async()'s docstring for the cost."
        )
        strict_batch_ordering = True

    def _checkpoint_save() -> None:
        if not incremental_save:
            return
        # utf-8-sig + atomic_to_csv: see the final save below for why.
        atomic_to_csv(dialogues_df, output_filename, index=False, encoding="utf-8-sig")

    # --- resume ---
    if resume:
        if not incremental_save:
            raise ValueError(
                "resume=True requires incremental_save=True -- there's no "
                "reliable partial output to resume FROM otherwise (without "
                "it, the output file is only ever written once, at the "
                "very end of a fully successful run)."
            )
        if not os.path.exists(output_filename):
            print(
                f"resume=True but no existing file at {output_filename} -- "
                "starting fresh (nothing to resume from)."
            )
        else:
            prior_df = pd.read_csv(output_filename)
            final_present_cols = [
                c for c in prior_df.columns if c.endswith("_final_present")
            ]
            if not final_present_cols or "dialogue_id" not in prior_df.columns or "turn_pair_index" not in prior_df.columns:
                raise ValueError(
                    f"Cannot resume from {output_filename}: it has no "
                    "per-model '*_final_present' columns (and/or is "
                    "missing dialogue_id/turn_pair_index), so it wasn't "
                    "produced with incremental_save, isn't a rated-dialogues "
                    "CSV from this pipeline, or predates resume support. "
                    "Move/rename it if you want to start a fresh run at "
                    "this same output path."
                )
            # Merge in only the per-model final-score columns (the ones
            # incremental_save actually checkpoints progressively, see
            # its docstring) -- matched by (dialogue_id, turn_pair_index),
            # which are stable/deterministic here (unlike generate.py,
            # rating.py doesn't sample anything itself; the row set comes
            # straight from dialogues_csv_path). A cue+model's column is
            # reused regardless of which --cue-group-config grouped it
            # under originally -- "{cue}_{model}_final_present" is the
            # same semantic quantity either way.
            merge_cols = ["dialogue_id", "turn_pair_index"] + final_present_cols
            dialogues_df = dialogues_df.merge(
                prior_df[merge_cols], on=["dialogue_id", "turn_pair_index"], how="left"
            )
            valid_mask = dialogues_df[final_present_cols].isin([0, 1])
            n_resumable_cells = int(valid_mask.sum().sum())
            n_retry_cells = int(
                dialogues_df[final_present_cols].notna().sum().sum() - n_resumable_cells
            )
            print(
                f"Resuming from {output_filename}: merged in "
                f"{len(final_present_cols)} previously-rated model/cue "
                f"column(s) covering {n_resumable_cells} already-VALIDLY-rated "
                "(row, cue, model) combination(s) -- these will be reused "
                "rather than re-rated. "
                + (
                    f"{n_retry_cells} combination(s) previously came back -1 "
                    "(couldn't parse a valid rating -- see the 'Format not "
                    "followed' warnings) and will be RE-ATTEMPTED, not reused, "
                    "since -1 isn't a valid result to lock in. "
                    if n_retry_cells
                    else ""
                )
                + "Note: a resumed row's raw-explanation/per-sample detail "
                "columns are replaced with a placeholder noting they were "
                "reused (only the final per-model score is checkpointed "
                "incrementally, not the per-sample detail behind it) -- the "
                "final score itself is exactly the original value, not "
                "approximated."
            )

    # rating loop
    #
    # session_budget_exceeded / session_budget_exceeded_message: set when
    # budget_guard's cap is hit partway through the LLM row loop below.
    # Rather than crashing or continuing to attempt calls that would
    # immediately fail, this cue unit still gets its columns written using
    # whatever models/rows completed (see the "calculate final cross-model
    # score" step below, which runs unconditionally), and then this flag
    # is checked once more at the end of the cue-unit loop body to skip
    # any remaining cue units entirely and go straight to saving the CSV.
    session_budget_exceeded = False
    session_budget_exceeded_message = None

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
                    "max_retries": classifier_max_retries,
                    "initial_backoff": classifier_initial_backoff,
                    "max_backoff": classifier_max_backoff,
                    # --------------
                }
                if classifier_max_timeout_retries is not None:
                    classifier_llm_config["max_timeout_retries"] = classifier_max_timeout_retries
                if budget_guard is not None:
                    classifier_llm_config["budget_guard"] = budget_guard
                if max_tokens_for_unit is not None:
                    classifier_llm_config["max_tokens"] = max_tokens_for_unit
                if classifier_timeout is not None:
                    classifier_llm_config["timeout"] = classifier_timeout
                if rate_limiter is not None:
                    # dict -> per-model instance (missing model = no
                    # limiting for it); single instance -> shared by every
                    # model. See this function's docstring.
                    model_rate_limiter = (
                        rate_limiter.get(model_name)
                        if isinstance(rate_limiter, dict)
                        else rate_limiter
                    )
                    if model_rate_limiter is not None:
                        classifier_llm_config["rate_limiter"] = model_rate_limiter
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

                def _rate_one_row(row):
                    """Exactly the original per-row body, extracted so it
                    can run either sequentially or concurrently (via
                    asyncio.to_thread) without duplicating logic. May
                    raise BudgetExceededError (never caught here --
                    handled by the caller, see below) or any other
                    exception from the classifier call (also never
                    caught here -- propagates and is fatal, same as
                    before this refactor)."""
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

                    return row_raw, row_processed, row_final

                def _rate_one_row_resumable(row):
                    """Wraps _rate_one_row: if resume=True and every cue
                    in this unit already has a VALID (0 or 1) value for
                    this model on this row (from a merged-in prior
                    checkpoint), reuses that value instead of re-rating --
                    see resume's docstring in this function for exactly
                    what "reuse" means for the raw-explanation/per-sample
                    columns (short version: the final score is exact, the
                    detail columns are a clearly-labeled placeholder).
                    -1 (couldn't parse a valid rating from the response --
                    see calculate_summary_stats' handling of this same
                    sentinel in analysis.py) is deliberately NOT treated
                    as already-done: it means the prior attempt failed to
                    produce a usable verdict, which is exactly the kind
                    of row resume should retry, not lock in as final."""
                    if resume:
                        col_names = [
                            f"{c}_{sanitized_model_name}_final_present" for c in cue_unit
                        ]
                        if all(
                            col in row.index
                            and pd.notna(row[col])
                            and int(row[col]) in (0, 1)
                            for col in col_names
                        ):
                            row_final = {
                                c: int(row[f"{c}_{sanitized_model_name}_final_present"])
                                for c in cue_unit
                            }
                            placeholder = (
                                "[reused from a resumed checkpoint -- original "
                                "per-sample detail not preserved]"
                            )
                            row_raw = {c: [placeholder] * num_samples for c in cue_unit}
                            row_processed = {
                                c: [row_final[c]] * num_samples for c in cue_unit
                            }
                            return row_raw, row_processed, row_final
                    return _rate_one_row(row)

                progress_desc = (
                    f"[unit {unit_idx + 1}/{len(call_units)}] {cue_unit} "
                    f"| model {model_idx + 1}/{len(classifier_models)} '{model_name}'"
                )

                row_items = list(dialogues_df.iterrows())

                # Ensure this model's per-cue "final" columns exist
                # before the row loop starts, so incremental
                # checkpointing can write into them as rows resolve.
                # pd.NA ("not yet attempted this run") is kept distinct
                # from a genuinely-computed -1 ("attempted, skipped/
                # invalid") -- only meaningful once incremental_save is
                # actually writing into these columns mid-loop; the
                # normal end-of-unit assignment below overwrites this
                # column for every row regardless, whether or not
                # incremental_save is on.
                if incremental_save:
                    for c in cue_unit:
                        col = f"{c}_{sanitized_model_name}_final_present"
                        if col not in dialogues_df.columns:
                            dialogues_df[col] = pd.NA

                def _write_partial_results(indexed_results, _cue_unit=cue_unit, _col_suffix=sanitized_model_name):
                    """indexed_results: list of (df_index, result) pairs.
                    Writes only this model's per-cue FINAL score for each
                    -- not the per-sample/raw-explanation columns, which
                    are still only written at this cue unit's normal
                    completion point below (see incremental_save's
                    docstring). Idempotent: rewriting an already-written
                    cell with the same value is harmless, but callers
                    should still only pass the newly-resolved slice each
                    time to avoid pointless repeated writes."""
                    for idx, result in indexed_results:
                        if isinstance(result, BudgetExceededError):
                            row_final = {c: -1 for c in _cue_unit}
                        else:
                            _row_raw, _row_processed, row_final = result
                        for c in _cue_unit:
                            dialogues_df.at[idx, f"{c}_{_col_suffix}_final_present"] = row_final[c]

                if max_concurrency <= 1:
                    # Sequential path: behaviorally identical to before
                    # this refactor, except a BudgetExceededError now
                    # stops the row loop (padding remaining rows as
                    # skipped) instead of propagating and crashing the
                    # whole `rate` command.
                    row_results = []
                    row_budget_exceeded_error = None
                    for idx, row in tqdm(row_items, desc=progress_desc, unit="turn"):
                        if row_budget_exceeded_error is not None:
                            result = row_budget_exceeded_error
                        else:
                            try:
                                result = _rate_one_row_resumable(row)
                            except BudgetExceededError as e:
                                row_budget_exceeded_error = e
                                result = e
                        row_results.append(result)
                        if incremental_save:
                            _write_partial_results([(idx, result)])
                            _checkpoint_save()
                elif strict_batch_ordering:
                    _last_written = [0]

                    def _on_chunk_complete(results_so_far):
                        if not incremental_save:
                            return
                        new_slice = list(
                            zip(
                                [idx for idx, _ in row_items[_last_written[0] : len(results_so_far)]],
                                results_so_far[_last_written[0]:],
                            )
                        )
                        _write_partial_results(new_slice)
                        _last_written[0] = len(results_so_far)
                        _checkpoint_save()

                    row_results = _rate_rows_chunked(
                        row_items,
                        _rate_one_row_resumable,
                        max_concurrency,
                        progress_desc,
                        on_chunk_complete=_on_chunk_complete,
                    )
                else:
                    row_results = _rate_rows_concurrently(
                        row_items, _rate_one_row_resumable, max_concurrency, progress_desc
                    )

                current_model_raw = {c: [] for c in cue_unit}
                current_model_processed = {c: [] for c in cue_unit}
                current_model_final = {c: [] for c in cue_unit}
                model_budget_exceeded_error = None
                for result in row_results:
                    if isinstance(result, BudgetExceededError):
                        model_budget_exceeded_error = model_budget_exceeded_error or result
                        for c in cue_unit:
                            current_model_raw[c].append(
                                ["Skipped - budget/iteration cap reached"] * num_samples
                            )
                            current_model_processed[c].append([-1] * num_samples)
                            current_model_final[c].append(-1)
                    else:
                        row_raw, row_processed, row_final = result
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

                if model_budget_exceeded_error is not None:
                    session_budget_exceeded = True
                    session_budget_exceeded_message = str(model_budget_exceeded_error)
                    print(
                        f"\nBudget/iteration cap reached while rating with "
                        f"model '{model_name}' on cue unit {cue_unit} "
                        f"({model_budget_exceeded_error}). This unit's "
                        "columns will still be written using whatever "
                        "models/rows completed; remaining classifier "
                        "model(s) for this unit and any remaining cue "
                        "unit(s) will not be attempted."
                    )
                    break  # stop the model loop for this cue unit


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

        if incremental_save:
            # This cue unit's columns are now fully written regardless
            # of incremental_save (see above) -- this just also
            # checkpoints that state to disk, on top of the finer
            # within-model checkpoints already taken during the row
            # loop(s) above.
            _checkpoint_save()

        if session_budget_exceeded:
            print(
                f"\nStopping after cue unit {unit_idx + 1}/{len(call_units)} "
                f"{cue_unit} ({session_budget_exceeded_message}); any "
                "remaining cue unit(s) will not be attempted. Saving "
                "everything rated so far."
            )
            break
    # end cue-unit loop

    try:
        # utf-8-sig, not plain utf-8 -- see generator.py's save_dialogues_to_csv
        # for why (Excel mojibake without a BOM; verified safe for this
        # codebase's own downstream reads either way). atomic_to_csv, not
        # dialogues_df.to_csv directly -- see io_utils.py; matters here
        # too since this same path may already have interim
        # incremental-save checkpoints written to it during the loop
        # above, which a direct write dying partway through would
        # corrupt.
        atomic_to_csv(dialogues_df, output_filename, index=False, encoding="utf-8-sig")
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
    classifier_timeout: float | None = None,
    classifier_max_retries: int = 5,
    classifier_max_timeout_retries: int | None = None,
    classifier_initial_backoff: float = 2.0,
    classifier_max_backoff: float = 60.0,
    rate_limiter: Union[RateLimiter, Dict[str, RateLimiter], None] = None,
    budget_guard: BudgetGuard | None = None,
    max_concurrency: int = 1,
    strict_batch_ordering: bool = False,
    incremental_save: bool = False,
    resume: bool = False,
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
        classifier_timeout, classifier_max_retries,
            classifier_max_timeout_retries, classifier_initial_backoff,
            classifier_max_backoff, rate_limiter, budget_guard,
            max_concurrency, strict_batch_ordering, incremental_save,
            resume: See
            rate_dialogues()'s docstring -- passed straight through.
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
        classifier_timeout=classifier_timeout,
        classifier_max_retries=classifier_max_retries,
        classifier_max_timeout_retries=classifier_max_timeout_retries,
        classifier_initial_backoff=classifier_initial_backoff,
        classifier_max_backoff=classifier_max_backoff,
        rate_limiter=rate_limiter,
        budget_guard=budget_guard,
        max_concurrency=max_concurrency,
        strict_batch_ordering=strict_batch_ordering,
        incremental_save=incremental_save,
        resume=resume,
        # --------------
        verbose=verbose,
    )
