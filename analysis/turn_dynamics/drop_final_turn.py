# Exploratory rerun (analysis plan, Section 10): the analysis without each
# dialogue's final rated turn. The linearity check showed that the last
# message of a dialogue - usually a closing exchange ("you're welcome") - has
# far fewer cues than the rest; this run asks whether the within-dialogue
# decrease remains without it. Not confirmatory: the pre-specified H1 result
# stands.
#
# Writes a copy of each rated CSV without the row of each dialogue's final
# rated turn, i.e. the last turn turn_dynamics.py would analyse: rated (at
# least one cue with a 0/1 rating), within --max-turns, and before the first
# empty reply. Every other row is copied unchanged, so turn numbers keep their
# values and each dialogue becomes one rated turn shorter; a dialogue with a
# single rated turn has none left. The input files are never changed.
#
# Usage:
#   python drop_final_turn.py --run label=path ... --out-dir normalized_drop_final

import argparse
import os

import pandas as pd

import turn_dynamics as td


def final_turn_rows(df: pd.DataFrame, max_turns: int) -> pd.Series:
    """True on the row of each dialogue's last analysable rated turn."""
    cue_cols = td.find_cue_columns(df)
    turn = df["turn_pair_index"].astype(int) + 1
    rated = df[cue_cols].isin([0, 1]).any(axis=1) & (turn <= max_turns)
    empty = td.find_empty_replies(df)
    first_empty = df["turn_pair_index"].where(empty).groupby(df["dialogue_id"]).transform("min")
    usable = rated & (first_empty.isna() | (df["turn_pair_index"] < first_empty))
    last = turn.where(usable).groupby(df["dialogue_id"]).transform("max")
    return usable & (turn == last)


def main() -> None:
    p = argparse.ArgumentParser(description="Copies of the rated CSVs without each dialogue's final rated turn.")
    p.add_argument("--run", action="append", required=True, help="label=path, as for turn_dynamics.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-turns", type=int, default=10)
    a = p.parse_args()
    for label, path in td.parse_run_specs(a.run):
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)  # written back exactly as read
        df = td.read_csv_strict(path)  # parsed as the analysis parses it
        if len(raw) != len(df):
            raise ValueError(f"{path}: the two readings differ in length.")
        df["model"] = label
        drop = final_turn_rows(df, a.max_turns)
        per_dialogue = drop.groupby(df["dialogue_id"]).sum()
        if per_dialogue.max() > 1:
            raise ValueError(f"{path}: more than one final turn in a dialogue.")
        rated_turns = (df[td.find_cue_columns(df)].isin([0, 1]).any(axis=1)).groupby(df["dialogue_id"]).sum()
        out_dir = os.path.join(a.out_dir, label)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(path))
        raw[~drop.to_numpy()].to_csv(out_path, index=False, encoding="utf-8")
        print(f"{label}: {int(drop.sum())} final turns dropped in {df['dialogue_id'].nunique()} dialogues; "
              f"{int((rated_turns == 1).sum())} dialogues had a single rated turn -> {out_path}")


if __name__ == "__main__":
    main()
