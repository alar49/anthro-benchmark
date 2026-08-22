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

import os
import json
import traceback
from typing import Dict, List, Tuple, Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm


pio.templates.default = "plotly_white"

# Must match generator.py's own _SUCCESSFUL_STATUSES exactly -- anything
# else in dialogue_status is either a budget/iteration-cap stop (status
# starts with "stopped_budget_exceeded_") or a genuine failure (status
# starts with "failed_at_turn_"). Duplicated here rather than imported to
# keep this module's only dependency on the rated CSV's columns, not on
# generator.py's internals.
_SUCCESSFUL_STATUSES = {"completed", "completed_early_natural_end"}

CATEGORY_MAPPING = {
    "internal states": ["desires", "emotions", "agency"],
    "personhood": [
        "personal history",
        "personal relationships",
        "sentience",
        "personal pronoun use", # it was "first-person prounoun use",
    ],
    "physical embodiment": [
        "physical embodiment",
        "movement and interactions", # it was "physical movement",
        "sensory input",
    ],
    "relationship building": [
        "empathy",
        "validation",
        "relatability",
        "explicit relationship status",
    ],
}


def load_data(
    rated_csv_path: str,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    """Loads rated data and identifies cue columns."""

    df = pd.read_csv(rated_csv_path)

    category_mapping = CATEGORY_MAPPING

    cue_present_columns = [
        col
        for col in df.columns
        if col.endswith("_present") and "_s" not in col and "_final_present" not in col
    ]

    if not cue_present_columns:
        raise ValueError(
            f"No '*_present' columns found in {rated_csv_path}. Cannot perform analysis."
        )

    tqdm.write(f"Found rating columns: {cue_present_columns}")

    all_mapped_cues = set(cue for cues in category_mapping.values() for cue in cues)
    found_cue_prefixes = set(col.replace("_present", "") for col in cue_present_columns)

    if not all_mapped_cues.issubset(found_cue_prefixes):
        missing_in_df = all_mapped_cues - found_cue_prefixes
        tqdm.write(
            f"Warning: Cues defined in mapping but not found in CSV's *_present columns: {missing_in_df}"
        )
    if not found_cue_prefixes.issubset(all_mapped_cues):
        missing_in_map = found_cue_prefixes - all_mapped_cues
        tqdm.write(
            f"Warning: Cues found in CSV (*_present columns) but not defined in category mapping: {missing_in_map}"
        )

    return df, category_mapping, cue_present_columns


def add_category_counts(
    df: pd.DataFrame,
    category_mapping: Dict[str, List[str]],
    cue_present_columns: List[str],
) -> pd.DataFrame:
    """Adds row-wise counts for each category based on cue presence."""
    tqdm.write("Adding row-wise category counts...")
    df_analysis = df.copy()
    found_cue_prefixes = {col.replace("_present", "") for col in cue_present_columns}

    for category, cues_in_category in category_mapping.items():
        category_col_name = f"{category}_count"
        relevant_cols = [
            f"{cue}_present" for cue in cues_in_category if cue in found_cue_prefixes
        ]

        if not relevant_cols:
            tqdm.write(
                f"  Skipping category '{category}': No corresponding '*_present' columns found."
            )
            df_analysis[category_col_name] = 0
            continue

        # sum only valid ratings (0 or 1), treat -1 as 0 for the sum
        df_analysis[category_col_name] = (
            df_analysis[relevant_cols]
            .map(lambda x: x if x in [0, 1] else 0)
            .sum(axis=1)
        )
        tqdm.write(f"  Added column: {category_col_name}")

    return df_analysis


def calculate_summary_stats(
    df: pd.DataFrame,
    category_mapping: Dict[str, List[str]],
    cue_present_columns: List[str],
) -> Dict[str, Any]:
    """Calculates overall cue percentages and category totals."""
    tqdm.write("Calculating summary statistics...")
    summary = {"cue_percentages": {}, "category_totals": {}}
    found_cue_prefixes = {col.replace("_present", "") for col in cue_present_columns}

    # percentages
    for col in cue_present_columns:
        cue_name = col.replace("_present", "")
        valid_ratings = df[col][df[col].isin([0, 1])]  # Filter out -1 (errors/skipped)
        if len(valid_ratings) == 0:
            percentage = 0.0
            tqdm.write(f"  Cue '{cue_name}': No valid ratings found.")
        else:
            percentage = (
                valid_ratings.sum() / len(valid_ratings)
            ) * 100  # Sum is count of 1s
        summary["cue_percentages"][cue_name] = round(percentage, 2)
        tqdm.write(
            f"  Cue '{cue_name}': {percentage:.2f}% present ({valid_ratings.sum()} / {len(valid_ratings)} valid turns)"
        )

    # category totals (sum of all '1's for cues in that category across all valid turns)
    for category, cues_in_category in category_mapping.items():
        category_total = 0
        relevant_cols = [
            f"{cue}_present" for cue in cues_in_category if cue in found_cue_prefixes
        ]
        if relevant_cols:
            # sum only 1s across all relevant columns and rows
            category_total = (
                df[relevant_cols].map(lambda x: 1 if x == 1 else 0).sum().sum()
            )
        summary["category_totals"][category] = int(
            category_total
        )  # ensure integer count
        tqdm.write(f"  Category '{category}': Total count = {category_total}")

    return summary


def calculate_missing_ratings_report(
    df: pd.DataFrame,
    cue_present_columns: List[str],
) -> Dict[str, Any]:
    """
    Counts "-1" cells (no valid rating produced -- see get_majority_vote
    in rating.py, and calculate_summary_stats' "Filter out -1
    (errors/skipped)" above) across the per-cue "*_present" aggregate
    columns identified by load_data(). A -1 here means the classifier
    never landed on a parseable 0/1 verdict for that (row, cue) pair --
    NOT that the cue was rated absent (that's a 0). Handy for eyeballing
    the effect of a --resume run: how many (row, cue) cells still need
    another pass, broken down by cue, and whether any turns are entirely
    unratable (most commonly an empty assistant_message -- e.g. a
    --stop-on-natural-end dialogue that ends on the user's turn; see
    rate_dialogues()'s "Skipped - Empty or invalid assistant message"
    case in rating.py, which is what drives every cue to -1 on such a
    row, including the regex-based "personal pronoun use" cue).

    Returns a dict with:
      - "n_rows": row count of df.
      - "n_cue_columns": how many cue columns were checked.
      - "all_cues_minus_one_rows": rows where EVERY cue column is -1.
      - "total_minus_one_cells": total -1 count, summed across every
        cue column and every row (i.e. over the whole cue x row matrix).
      - "total_cells": n_rows * n_cue_columns, for context/denominators.
      - "minus_one_per_cue": {cue_name: count}, one entry per cue,
        in the same order as cue_present_columns.
    """
    tqdm.write("Calculating missing-ratings (-1) report...")
    cue_name_by_col = {col: col.replace("_present", "") for col in cue_present_columns}

    is_minus_one = df[cue_present_columns] == -1

    all_cues_minus_one_rows = int(is_minus_one.all(axis=1).sum())
    total_minus_one_cells = int(is_minus_one.sum().sum())
    total_cells = int(len(df) * len(cue_present_columns))
    minus_one_per_cue = {
        cue_name_by_col[col]: int(is_minus_one[col].sum()) for col in cue_present_columns
    }

    report = {
        "n_rows": int(len(df)),
        "n_cue_columns": len(cue_present_columns),
        "all_cues_minus_one_rows": all_cues_minus_one_rows,
        "total_minus_one_cells": total_minus_one_cells,
        "total_cells": total_cells,
        "minus_one_per_cue": minus_one_per_cue,
    }

    tqdm.write(
        f"  Rows where EVERY cue is -1: {all_cues_minus_one_rows} / {len(df)}"
    )
    tqdm.write(
        f"  Total -1 cells across all cues: {total_minus_one_cells} / {total_cells}"
    )
    tqdm.write("  -1 count per cue:")
    for cue_name, count in minus_one_per_cue.items():
        tqdm.write(f"    '{cue_name}': {count} / {len(df)} rows")

    return report


def filter_to_first_n_turns(
    df: pd.DataFrame,
    n: int,
    turn_col: str = "turn_pair_index",
    require_min_turns: bool = False,
) -> pd.DataFrame:
    """
    Restricts to each dialogue's first n turn-pairs (0-indexed
    turn_col < n) -- for a fixed-length comparison against prior work
    that used a fixed number of turns, rather than the variable-length
    dialogues --stop-on-natural-end can produce here.

    require_min_turns (default False): if False, a dialogue that
    naturally ended before reaching n turns keeps whatever it has (fewer
    than n rows) rather than being dropped entirely -- see
    run_analysis()'s docstring for why that's the default. If True,
    dialogues whose total length is below n are excluded ENTIRELY first,
    so every dialogue that survives contributes exactly n turns, not
    fewer -- "total length" here means the dialogue's full row count in
    df as passed in, so call this on the unfiltered/full dataset, not an
    already-windowed one, or the min-turns check would be comparing
    against an already-truncated length instead of each dialogue's real
    total.
    """
    if turn_col not in df.columns:
        raise ValueError(
            f"Expected a '{turn_col}' column to filter to the first {n} turns."
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    if require_min_turns:
        if "dialogue_id" not in df.columns:
            raise ValueError(
                "Expected a 'dialogue_id' column to apply require_min_turns."
            )
        max_turn_per_dialogue = df.groupby("dialogue_id")[turn_col].transform("max")
        # A dialogue whose highest turn_pair_index is >= n-1 has AT LEAST
        # n turn-pairs present (indices 0..n-1 all exist for it).
        df = df[max_turn_per_dialogue >= n - 1]

    return df[df[turn_col] < n].copy()


def write_dialogue_length_report(df: pd.DataFrame, output_path: str) -> None:
    """
    Writes a plain-text report on dialogue completion and length,
    computed from the FULL/unfiltered rated dataframe -- i.e. before any
    first-N-turns windowing, since this describes the shape of the
    underlying dataset as a whole, not one slice of it.

    "Length" here = row count per dialogue_id, the same quantity
    filter_to_first_n_turns()'s turn_col < n check effectively counts
    (for n <= a dialogue's length) -- so a dialogue reported here as
    length 5 is exactly one that require_min_turns=True, n=5 would
    include.

    dialogue_status is bucketed via _SUCCESSFUL_STATUSES (see its
    docstring): completed / budget-capped / failed are reported as three
    SEPARATE buckets rather than a single completed-vs-failed split,
    since a budget or iteration cap being hit isn't evidence anything
    went wrong with that specific dialogue -- it's an external session
    limit, not a per-dialogue failure.
    """
    if "dialogue_id" not in df.columns:
        raise ValueError(
            "Expected a 'dialogue_id' column to build the dialogue length report."
        )
    if "turn_pair_index" not in df.columns:
        raise ValueError(
            "Expected a 'turn_pair_index' column to build the dialogue length report."
        )

    lengths = df.groupby("dialogue_id").size()
    total_dialogues = int(len(lengths))

    lines: List[str] = []
    lines.append("=== Dialogue Length & Completion Report ===")
    lines.append("")
    lines.append("--- Completion Summary ---")
    lines.append(f"Total dialogues in CSV: {total_dialogues}")

    if "dialogue_status" in df.columns and total_dialogues:
        status_per_dialogue = df.groupby("dialogue_id")["dialogue_status"].first()

        def _bucket(status: Any) -> str:
            if pd.isna(status):
                return "unknown"
            s = str(status)
            if s in _SUCCESSFUL_STATUSES:
                return "completed"
            if s.startswith("stopped_budget_exceeded_"):
                return "budget_capped"
            if s.startswith("failed_at_turn_"):
                return "failed"
            return "unknown"

        buckets = status_per_dialogue.map(_bucket)
        n_natural_end = int((status_per_dialogue == "completed_early_natural_end").sum())
        n_completed = int((buckets == "completed").sum())
        n_full_length = n_completed - n_natural_end
        n_budget_capped = int((buckets == "budget_capped").sum())
        n_failed = int((buckets == "failed").sum())
        n_unknown = int((buckets == "unknown").sum())

        def _pct(x: int) -> str:
            return f"{x / total_dialogues * 100:.2f}%"

        lines.append(f"  Successfully completed: {n_completed} ({_pct(n_completed)})")
        lines.append(f"    - completed (reached full requested length): {n_full_length}")
        lines.append(f"    - completed_early_natural_end: {n_natural_end}")
        lines.append(
            f"  Stopped early (budget/iteration cap reached): {n_budget_capped} ({_pct(n_budget_capped)})"
        )
        lines.append(f"  Failed (LLM/generation error): {n_failed} ({_pct(n_failed)})")
        if n_unknown:
            lines.append(f"  Unrecognized dialogue_status value: {n_unknown} ({_pct(n_unknown)})")
    else:
        lines.append("  (no 'dialogue_status' column found -- completion breakdown skipped)")
    lines.append("")

    lines.append("--- Dialogue Length Distribution (rows per dialogue_id) ---")
    if "user_message" in df.columns and "assistant_message" in df.columns:
        is_blank = (
            df["user_message"].isna() | (df["user_message"].astype(str).str.strip() == "")
        ) & (
            df["assistant_message"].isna()
            | (df["assistant_message"].astype(str).str.strip() == "")
        )
        blank_row_dialogue_count = int(df.loc[is_blank, "dialogue_id"].nunique())
        if blank_row_dialogue_count:
            lines.append(
                f"Note: {blank_row_dialogue_count} of these dialogues include a trailing "
                f"row with no message content at all (the natural-end 'sentinel-only "
                f"closing reply' artifact) -- it still counts as +1 toward that "
                f"dialogue's length below."
            )
            lines.append("")

    if total_dialogues:
        length_counts = lengths.value_counts().sort_index()
        for length, count in length_counts.items():
            lines.append(f"  {int(length)}-turn dialogues: {int(count)}")
    else:
        lines.append("  (no dialogues found)")
    lines.append("")

    lines.append("--- Length Statistics ---")
    if total_dialogues:
        modes = lengths.mode().tolist()
        mode_str = ", ".join(str(int(m)) for m in modes)
        # ddof=1 (sample std/var, pandas' default) -- called out explicitly
        # since sample vs. population changes the number and this is
        # meant to be read precisely, not guessed at.
        std_dev = float(lengths.std()) if total_dialogues > 1 else 0.0
        variance = float(lengths.var()) if total_dialogues > 1 else 0.0
        lines.append(f"  Count: {total_dialogues}")
        lines.append(f"  Mean: {float(lengths.mean()):.3f}")
        lines.append(f"  Median: {float(lengths.median())}")
        lines.append(f"  Mode: {mode_str}" + (" (tied)" if len(modes) > 1 else ""))
        lines.append(f"  Sample Std Dev (ddof=1): {std_dev:.3f}")
        lines.append(f"  Sample Variance (ddof=1): {variance:.3f}")
        lines.append(f"  Min: {int(lengths.min())}")
        lines.append(f"  Max: {int(lengths.max())}")
    else:
        lines.append("  (no dialogues found)")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    tqdm.write(f"Saved dialogue length & completion report to: {output_path}")


def calculate_per_dialogue_stats(
    df: pd.DataFrame,
    category_mapping: Dict[str, List[str]],
    cue_present_columns: List[str],
) -> pd.DataFrame:
    """
    Same cue-percentage / category-total definitions as
    calculate_summary_stats(), computed separately per dialogue_id
    instead of pooled across the whole dataset -- for spotting individual
    dialogues that are outliers rather than only ever seeing the
    dataset-wide average.

    One row per dialogue_id. Turn counts reflect however many rows for
    that dialogue_id are present in `df` -- i.e. if df has already been
    filtered (see filter_to_first_n_turns), the returned stats are scoped
    to that filtered window, not each dialogue's full length.

    Note one deliberate difference from calculate_summary_stats: when a
    dialogue has zero validly-rated turns for a given cue, its "_pct"
    column here is left as None (NaN) rather than 0.0. At the pooled/
    overall level, "no valid ratings at all" is rare enough that folding
    it into 0% barely matters; at the per-dialogue level it's common
    (short dialogues, or every rating for one cue happening to error
    out), and 0.0% there would misleadingly read as "measured absent"
    rather than "no data" -- the whole point of a per-dialogue table
    being for close inspection. "_present_count" and "_valid_turns" are
    always populated (both 0 when there's no data) so this is fully
    visible either way, not just hidden behind a blank cell.
    """
    if "dialogue_id" not in df.columns:
        raise ValueError("Expected a 'dialogue_id' column to compute per-dialogue stats.")

    found_cue_prefixes = {col.replace("_present", "") for col in cue_present_columns}

    # Constant-per-dialogue metadata, carried through so this table is
    # self-contained for filtering/pivoting without re-joining back to
    # the raw rated CSV. Only included if actually present in df (this
    # function doesn't assume a specific upstream CSV schema beyond
    # dialogue_id + the cue columns).
    passthrough_cols = [
        c
        for c in [
            "prompt_category",
            "prompt_cue",
            "user_llm",
            "target_llm",
            "reasoning_mode",
            "reasoning_effort",
            "dialogue_status",
        ]
        if c in df.columns
    ]

    rows = []
    for dialogue_id, group in df.groupby("dialogue_id", sort=False):
        row: Dict[str, Any] = {
            "dialogue_id": dialogue_id,
            "n_turns_in_window": len(group),
        }

        for c in passthrough_cols:
            # Constant per dialogue by construction (generator.py writes
            # the same value on every row of a dialogue) -- take the
            # first non-null occurrence rather than assuming row 0 is
            # always populated.
            non_null = group[c].dropna()
            row[c] = non_null.iloc[0] if len(non_null) else None

        for col in cue_present_columns:
            cue_name = col.replace("_present", "")
            valid = group[col][group[col].isin([0, 1])]
            present_count = int(valid.sum()) if len(valid) else 0
            row[f"{cue_name}_present_count"] = present_count
            row[f"{cue_name}_valid_turns"] = int(len(valid))
            row[f"{cue_name}_pct"] = (
                round(present_count / len(valid) * 100, 2) if len(valid) else None
            )

        for category, cues_in_category in category_mapping.items():
            relevant_cols = [
                f"{cue}_present" for cue in cues_in_category if cue in found_cue_prefixes
            ]
            category_total = 0
            if relevant_cols:
                category_total = int(
                    group[relevant_cols].map(lambda x: 1 if x == 1 else 0).sum().sum()
                )
            row[f"{category}_total"] = category_total

        rows.append(row)

    return pd.DataFrame(rows)


def plot_cue_percentages(
    cue_percentages: Dict[str, float],
    output_dir: str,
    filename_suffix: str = "",
    title_suffix: str = "",
):
    """Creates a bar chart of cue percentages. filename_suffix is
    inserted before the file extension (e.g. "_first_5_turns") --
    empty by default, which preserves the original "cue_percentages.png"
    filename exactly. title_suffix is appended to the plot title."""
    if not cue_percentages:
        tqdm.write("No cue percentages to plot.")
        return

    tqdm.write("Generating cue percentages bar chart...")
    cues = list(cue_percentages.keys())
    percentages = list(cue_percentages.values())

    fig = go.Figure(
        [go.Bar(x=cues, y=percentages, text=percentages, textposition="auto")]
    )
    fig.update_layout(
        title=f"Percentage of messages where behavior occurs{title_suffix}",
        xaxis_title="Behavior",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
    )

    plot_path_png = os.path.join(output_dir, f"cue_percentages{filename_suffix}.png")
    plot_path_html = os.path.join(output_dir, f"cue_percentages{filename_suffix}.html")

    try:
        # try to save PNG first
        try:
            fig.write_image(plot_path_png)
            tqdm.write(f"  Saved plot to: {plot_path_png}")
        except Exception as e:
            tqdm.write(f"  Error saving PNG plot: {e}")
            tqdm.write(
                '  To save PNGs, install kaleido: pip install -U "kaleido>=0.1.0,<0.2.0"'
            )

            # fallback to HTML if PNG fails
            fig.write_html(plot_path_html)
            tqdm.write(f"  Saved HTML plot to: {plot_path_html}")
    except Exception as e:
        tqdm.write(f"  Error saving plot: {e}")


def plot_category_radar(
    category_totals: Dict[str, int],
    output_dir: str,
    filename_suffix: str = "",
    title_suffix: str = "",
):
    """Creates a radar chart of category totals. filename_suffix is
    inserted before the file extension -- empty by default, which
    preserves the original "category_radar.png" filename exactly.
    title_suffix is appended to the plot title."""
    if not category_totals or len(category_totals) < 3:
        tqdm.write(
            f"Skipping radar plot: Need at least 3 categories with totals, found {len(category_totals)}."
        )
        return

    tqdm.write("Generating category totals radar chart...")
    categories = list(category_totals.keys())
    totals = list(category_totals.values())

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=totals + [totals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Total Cue Counts",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(totals) * 1.1 if totals else 1])
        ),
        showlegend=False,
        title=f"Total counts of behaviors per category{title_suffix}",
    )

    plot_path_png = os.path.join(output_dir, f"category_radar{filename_suffix}.png")
    plot_path_html = os.path.join(output_dir, f"category_radar{filename_suffix}.html")

    try:
        try:
            fig.write_image(plot_path_png)
            tqdm.write(f"  Saved plot to: {plot_path_png}")
        except Exception as e:
            tqdm.write(f"  Error saving PNG plot: {e}")
            tqdm.write(
                '  To save PNGs, install kaleido: pip install -U "kaleido>=0.1.0,<0.2.0"'
            )

            # fallback to HTML if PNG fails
            fig.write_html(plot_path_html)
            tqdm.write(f"  Saved HTML plot to: {plot_path_html}")
    except Exception as e:
        tqdm.write(f"  Error saving plot: {e}")


def run_analysis(
    rated_csv_path: str,
    output_dir: str = "analysis_results",
    category_mapping_path: str = None,
    first_n_turns: int = 5,
    require_min_turns: bool = False,
):
    """
    Main function to run the analysis pipeline.

    Produces two parallel sets of results:
      - "final": every rated turn, whatever length each dialogue actually
        reached. Unchanged from this function's original behavior --
        same filenames as before (analysis_with_categories.csv,
        summary_stats.json, cue_percentages.png, category_radar.png),
        for backward compatibility with anything already reading them.
      - "first_{first_n_turns}_turns" (or "..._strict" if
        require_min_turns=True): the same statistics restricted to each
        dialogue's first `first_n_turns` turn-pairs (turn_pair_index <
        first_n_turns), for a fixed-length comparison against prior work
        that didn't have variable-length dialogues.
          - require_min_turns=False (default): a dialogue that ended
            earlier than first_n_turns (e.g. via --stop-on-natural-end)
            contributes whatever turns it has, rather than being
            excluded outright -- so a 3-turn dialogue is analyzed
            identically in both the "final" and "first_n_turns" views.
          - require_min_turns=True: dialogues shorter than first_n_turns
            are excluded ENTIRELY from the first_n_turns view (but still
            included in "final"), so every dialogue contributing to it
            has exactly first_n_turns turns, not fewer -- for a stricter,
            fixed-length-only comparison. The "_strict" filename suffix
            keeps this from overwriting the non-strict run's output if
            you compare both against the same output_dir.

    Both sets are computed at two granularities:
      - overall: pooled across the whole dataset (as before) --
        cue_percentages.json / cue_percentages.png / category_radar.png
        and their windowed counterparts.
      - per-dialogue: one row per dialogue_id, in per_dialogue_stats*.csv
        -- for spotting individual dialogues that are outliers rather
        than only ever seeing the dataset-wide average. See
        calculate_per_dialogue_stats()'s docstring for exactly what each
        column means and one deliberate difference from the overall
        stats' handling of "no valid ratings".

    Also writes dialogue_length_report.txt: a plain-text breakdown of
    dialogue completion status and length distribution across the WHOLE
    dataset (not windowed) -- see write_dialogue_length_report()'s
    docstring for exactly what it reports and how completion is
    bucketed.
    """
    tqdm.write("\n--- Starting Analysis ---")
    tqdm.write(f"Rated CSV: {rated_csv_path}")
    tqdm.write(f"Output Directory: {output_dir}")
    tqdm.write("Using hardcoded category mapping")
    tqdm.write(
        f"'First N turns' window: first_n_turns={first_n_turns}, "
        f"require_min_turns={require_min_turns}"
    )

    window_label = f"first_{first_n_turns}_turns"
    if require_min_turns:
        window_label += "_strict"

    stages = [
        "Loading rated data",
        "Adding category counts (final)",
        "Calculating summary statistics (final)",
        "Calculating missing-ratings (-1) report",
        "Calculating per-dialogue statistics (final)",
        "Writing dialogue length report",
        f"Filtering to {window_label}",
        f"Calculating summary statistics ({window_label})",
        f"Calculating per-dialogue statistics ({window_label})",
        "Plotting cue percentages (final)",
        "Plotting category radar (final)",
        f"Plotting cue percentages ({window_label})",
        f"Plotting category radar ({window_label})",
        "Saving enhanced CSVs",
        "Saving per-dialogue CSVs",
        "Saving summary JSONs",
    ]

    try:
        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        with tqdm(total=len(stages), desc="Analysis", unit="step") as pbar:
            pbar.set_description(stages[0])
            df, category_mapping, cue_present_columns = load_data(rated_csv_path)
            pbar.update(1)

            pbar.set_description(stages[1])
            df_final = add_category_counts(df, category_mapping, cue_present_columns)
            pbar.update(1)

            pbar.set_description(stages[2])
            summary_stats_final = calculate_summary_stats(
                df_final, category_mapping, cue_present_columns
            )
            pbar.update(1)

            pbar.set_description(stages[3])
            # Computed on the FULL (final, unwindowed) data -- the
            # windowed "first_n_turns" view is a different, orthogonal
            # slicing concept and isn't what you want when checking a
            # --resume run's coverage over the whole rated CSV.
            missing_ratings_report = calculate_missing_ratings_report(
                df_final, cue_present_columns
            )
            pbar.update(1)

            pbar.set_description(stages[4])
            per_dialogue_final = calculate_per_dialogue_stats(
                df_final, category_mapping, cue_present_columns
            )
            pbar.update(1)

            pbar.set_description(stages[5])
            length_report_path = os.path.join(output_dir, "dialogue_length_report.txt")
            write_dialogue_length_report(df_final, length_report_path)
            pbar.update(1)

            pbar.set_description(stages[6])
            df_window = filter_to_first_n_turns(
                df_final, first_n_turns, require_min_turns=require_min_turns
            )
            tqdm.write(
                f"  {window_label}: {len(df_window)} of {len(df_final)} rated turns "
                f"kept, across {df_window['dialogue_id'].nunique()} of "
                f"{df_final['dialogue_id'].nunique()} dialogues."
            )
            pbar.update(1)

            pbar.set_description(stages[7])
            summary_stats_window = calculate_summary_stats(
                df_window, category_mapping, cue_present_columns
            )
            pbar.update(1)

            pbar.set_description(stages[8])
            per_dialogue_window = calculate_per_dialogue_stats(
                df_window, category_mapping, cue_present_columns
            )
            pbar.update(1)

            pbar.set_description(stages[9])
            plot_cue_percentages(summary_stats_final["cue_percentages"], output_dir)
            pbar.update(1)

            pbar.set_description(stages[10])
            plot_category_radar(summary_stats_final["category_totals"], output_dir)
            pbar.update(1)

            pbar.set_description(stages[11])
            plot_cue_percentages(
                summary_stats_window["cue_percentages"],
                output_dir,
                filename_suffix=f"_{window_label}",
                title_suffix=f" (first {first_n_turns} turns)",
            )
            pbar.update(1)

            pbar.set_description(stages[12])
            plot_category_radar(
                summary_stats_window["category_totals"],
                output_dir,
                filename_suffix=f"_{window_label}",
                title_suffix=f" (first {first_n_turns} turns)",
            )
            pbar.update(1)

            pbar.set_description(stages[13])
            enhanced_csv_path = os.path.join(output_dir, "analysis_with_categories.csv")
            df_final.to_csv(enhanced_csv_path, index=False, encoding="utf-8-sig")
            tqdm.write(f"Saved enhanced dataframe (final) to: {enhanced_csv_path}")
            pbar.update(1)

            pbar.set_description(stages[14])
            per_dialogue_final_path = os.path.join(
                output_dir, "per_dialogue_stats.csv"
            )
            per_dialogue_final.to_csv(per_dialogue_final_path, index=False, encoding="utf-8-sig")
            tqdm.write(f"Saved per-dialogue stats (final) to: {per_dialogue_final_path}")

            per_dialogue_window_path = os.path.join(
                output_dir, f"per_dialogue_stats_{window_label}.csv"
            )
            per_dialogue_window.to_csv(per_dialogue_window_path, index=False, encoding="utf-8-sig")
            tqdm.write(
                f"Saved per-dialogue stats ({window_label}) to: {per_dialogue_window_path}"
            )
            pbar.update(1)

            pbar.set_description(stages[15])
            summary_json_path = os.path.join(output_dir, "summary_stats.json")
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary_stats_final, f, indent=2)
            tqdm.write(f"Saved summary statistics (final) to: {summary_json_path}")

            summary_json_window_path = os.path.join(
                output_dir, f"summary_stats_{window_label}.json"
            )
            with open(summary_json_window_path, "w", encoding="utf-8") as f:
                json.dump(summary_stats_window, f, indent=2)
            tqdm.write(
                f"Saved summary statistics ({window_label}) to: {summary_json_window_path}"
            )

            missing_ratings_json_path = os.path.join(
                output_dir, "missing_ratings_report.json"
            )
            with open(missing_ratings_json_path, "w", encoding="utf-8") as f:
                json.dump(missing_ratings_report, f, indent=2)
            tqdm.write(
                f"Saved missing-ratings (-1) report to: {missing_ratings_json_path}"
            )
            pbar.update(1)

        tqdm.write("\n--- Analysis Complete ---")
        tqdm.write(f"All results saved to: {os.path.abspath(output_dir)}")

    except Exception as e:
        tqdm.write(f"Error during analysis: {e}")
        traceback.print_exc()
