"""
Extract a randomized, stratified-balanced sample of first_turns.csv.

This is a corrected version of the sampling script. See the accompanying
notes for exactly what was wrong with the original version and why each
change below was made -- nothing here is a stylistic preference, every
change fixes a behavior that was verified to be broken or misleading.
"""

import os
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
number_of_entries_per_combination: int = 2
RANDOM_SEED: int = 42
ALLOW_REPLACEMENT: bool = False

# Path to the dataset. Defaults to the copy bundled inside the installed
# anthro_benchmark package (anthro_benchmark/prompt_sets/first_turns.csv) --
# the exact same file DialogueGenerator._load_prompts() reads via
# importlib.resources when no --custom-prompt-csv is given. Resolving it
# the same way here guarantees the sample is drawn from whatever your
# local checkout actually contains, rather than a URL that could silently
# diverge from it. Only pip installs anthro-benchmark in editable mode
# (`pip install -e .` from the repo root) for this auto-resolution to
# work; otherwise edit the fallback path below to point at your checkout.
try:
    import importlib.resources
    DATASET_PATH: str = str(
        importlib.resources.files("anthro_benchmark.prompt_sets") / "first_turns.csv"
    )
except Exception:
    # Fallback if the package isn't installed -- point this at your
    # local checkout, e.g.:
    DATASET_PATH: str = "anthro_benchmark/prompt_sets/first_turns.csv"
OUTPUT_CSV_PATH: str = "balanced_first_turns_sample.csv"

CONDITION_COLUMNS = [
    "use_domain",
    "use_scenario",
    "empathy",
    "professionalism",
    "cue",
    "behavior_category",
]


def sample_one_group(group: pd.DataFrame, n_requested: int, allow_replacement: bool, seed: int) -> pd.DataFrame:
    """Sample n_requested rows from a single combination's rows.

    - If the group has enough rows, always sample WITHOUT replacement
      (replacement is never needed here, so it's never used, regardless
      of the allow_replacement flag).
    - If the group is short, either take everything available (default)
      or sample WITH replacement to pad up to n_requested, but only if
      allow_replacement was explicitly set.
    """
    available_n = len(group)
    if available_n >= n_requested:
        return group.sample(n=n_requested, replace=False, random_state=seed)
    if allow_replacement:
        return group.sample(n=n_requested, replace=True, random_state=seed)
    return group  # take all available rows, no replacement


def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    missing_cols = [col for col in CONDITION_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    print("\n--- Original Distribution Across Condition Combinations ---")
    combination_counts = (
        df.groupby(CONDITION_COLUMNS, dropna=False)
        .size()
        .reset_index(name="original_count")
    )
    print(f"Total unique condition combinations found: {len(combination_counts)}")

    undersampled = combination_counts[
        combination_counts["original_count"] < number_of_entries_per_combination
    ]
    if not undersampled.empty:
        print(
            f"\n[Warning] {len(undersampled)} combination(s) have fewer than "
            f"{number_of_entries_per_combination} entries available."
        )

    # --- Stratified sampling, done via index collection rather than
    # groupby(...).apply(...). This sidesteps a real bug: as of pandas
    # 2.2 the "include_groups" behavior around groupby-apply started
    # changing, and as of pandas 3.0 (Jan 2026) DataFrameGroupBy.apply()
    # excludes the grouping columns from what's handed to (and returned
    # by) the applied function by default. Concretely, running the
    # original groupby(...).apply(sample_group) pattern on pandas 3.0.x
    # silently drops use_domain/use_scenario/empathy/professionalism/
    # cue/behavior_category from the output -- i.e. exactly the columns
    # that make this a *stratified* sample, and that the generator needs
    # for cue/category metadata. Collecting index labels and slicing the
    # original df with .loc[] avoids the whole issue and works
    # identically on old and new pandas.
    sampled_index_labels = []
    group_position = 0
    for _, group in df.groupby(CONDITION_COLUMNS, sort=True, dropna=False):
        # Each group gets its own derived seed (RANDOM_SEED + position in
        # the stable, sorted group order) instead of reusing RANDOM_SEED
        # unmodified for every group. Reusing the same seed for every
        # group means groups of equal size always draw the same
        # *relative* row positions -- verified on this dataset: every
        # one of the 96 combinations has exactly 10 rows, and sampling
        # each with random_state=42 picks relative positions (1, 8) in
        # literally all 96 of them, every time. Since the 10 rows within
        # a combination are 10 different prompt phrasings
        # (original_prompt), that collapse means only 2 of the 10
        # phrasings would ever appear in the sample, for every single
        # combination -- not what "randomized" is meant to deliver.
        group_seed = RANDOM_SEED + group_position
        sampled_group = sample_one_group(
            group, number_of_entries_per_combination, ALLOW_REPLACEMENT, group_seed
        )
        sampled_index_labels.extend(sampled_group.index.tolist())
        group_position += 1

    sampled_df = df.loc[sampled_index_labels].copy()

    # Shuffle row order across combinations (not just within combinations).
    # groupby() with sort=True (the default) yields groups in a stable,
    # sorted order, so without this step every row from the first
    # combination(s) would end up at the top of the CSV and the file
    # would stay "blocked" by combination. That matters downstream: the
    # CLI's DialogueGenerator walks self.prompts in file order (row 0,
    # row 1, ...). If a generation run gets cut short (rate limit,
    # budget guard, crash, manual stop), only the first few combinations
    # would ever get generated, defeating the point of a *balanced*
    # sample for any partial run.
    sampled_df = sampled_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # --- Sanity check: verify the output actually has every condition
    # column before writing anything to disk. This is the exact
    # regression check that would have caught the pandas 3.0 bug above.
    missing_after_sampling = [c for c in CONDITION_COLUMNS if c not in sampled_df.columns]
    if missing_after_sampling:
        raise RuntimeError(
            f"Sampling silently dropped columns: {missing_after_sampling}. "
            "Do not write the output file -- inspect the sampling logic."
        )

    print("\n--- Sampling Complete ---")
    print(f"Requested entries per combination: {number_of_entries_per_combination}")
    print(f"Total rows in balanced output dataset: {len(sampled_df)}")

    # Verify achieved balance per combination
    achieved_counts = (
        sampled_df.groupby(CONDITION_COLUMNS, dropna=False)
        .size()
        .reset_index(name="achieved_count")
    )
    n_short = (achieved_counts["achieved_count"] < number_of_entries_per_combination).sum()
    print(f"Combinations achieving the full requested count: {len(achieved_counts) - n_short} / {len(achieved_counts)}")

    sampled_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Sampled dataset saved to: {OUTPUT_CSV_PATH}")
    return sampled_df


if __name__ == "__main__":
    main()
