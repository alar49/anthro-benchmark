"""
Compares the current one-cue-per-call baseline (classifiers.LLMClassifier,
as used in rating.py's rate_dialogues loop) against a candidate cue grouping
(cue_grouping.LLMGroupClassifier), on the same pilot dialogue set, and
reports per-cue label agreement so you can decide whether a grouping is
safe to promote to production before running it on the full dataset.

Usage:
    python test_cue_grouping_harness.py --dialogues pilot_dialogues.csv \
        --config A_interference_avoidant --classifier-model <model_name>

pilot_dialogues.csv needs at least user_message and assistant_message
columns (same schema rate_dialogues() expects).

Cost note: raw call-count reduction is a proxy, not the exact dollar
saving -- the grouped prompt is longer (N definitions+examples instead of
1) and the response is longer too (N reasons instead of 1). The saving
comes specifically from the turn text (user_message + assistant_message,
often the dominant length) being sent once per group instead of once per
cue. If your turns are short relative to the definitions/examples block,
the saving will be smaller than the raw call-count ratio suggests --
worth checking on your actual data rather than assuming.
"""

import argparse

import pandas as pd

from anthro_benchmark.classifier.classifiers import LLMClassifier
from anthro_benchmark.classifier.cue_definitions import CUE_DEFINITIONS
from anthro_benchmark.classifier.cue_grouping import (
    ALL_LLM_CUES,
    CUE_GROUP_CONFIGS,
    LLMGroupClassifier,
)


def run_baseline(df: pd.DataFrame, classifier_llm_config: dict) -> dict:
    """One LLMClassifier per cue, called once per turn -- exactly what
    rate_dialogues() does today. 13 calls/turn."""
    results = {cue: [] for cue in ALL_LLM_CUES}
    classifiers = {
        cue: LLMClassifier(
            classifier_llm_config,
            cue,
            CUE_DEFINITIONS[cue]["definition"],
            CUE_DEFINITIONS[cue].get("examples", []),
        )
        for cue in ALL_LLM_CUES
    }
    for _, row in df.iterrows():
        for cue in ALL_LLM_CUES:
            score, _ = classifiers[cue].rate_turn_messages(
                cue=cue,
                assistant_turn_message=row["assistant_message"],
                user_turn_message=row["user_message"],
            )
            results[cue].append(score)
    return results


def run_grouped(df: pd.DataFrame, classifier_llm_config: dict, groups: list) -> dict:
    """One LLMGroupClassifier per group, called once per turn. len(groups) calls/turn."""
    results = {cue: [] for group in groups for cue in group}
    group_classifiers = [LLMGroupClassifier(classifier_llm_config, group) for group in groups]
    for _, row in df.iterrows():
        for gc in group_classifiers:
            scores = gc.rate_turn_messages(row["assistant_message"], row["user_message"])
            for cue, (score, _) in scores.items():
                results[cue].append(score)
    return results


def compare(baseline: dict, grouped: dict) -> pd.DataFrame:
    rows = []
    for cue in ALL_LLM_CUES:
        b, g = baseline[cue], grouped[cue]
        n = len(b)
        agree = sum(1 for x, y in zip(b, g) if x == y)
        rows.append(
            {
                "cue": cue,
                "n": n,
                "raw_agreement": agree / n if n else float("nan"),
                "baseline_yes_rate": (sum(1 for x in b if x == 1) / n) if n else float("nan"),
                "grouped_yes_rate": (sum(1 for x in g if x == 1) / n) if n else float("nan"),
                "baseline_parse_errors": sum(1 for x in b if x == -1),
                "grouped_parse_errors": sum(1 for x in g if x == -1),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", required=True, help="Pilot dialogues CSV (user_message, assistant_message columns)")
    parser.add_argument("--config", required=True, choices=list(CUE_GROUP_CONFIGS.keys()))
    parser.add_argument("--classifier-model", required=True)
    parser.add_argument("--out", default="cue_grouping_comparison.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.dialogues)
    missing = {"user_message", "assistant_message"} - set(df.columns)
    if missing:
        raise ValueError(f"dialogues CSV missing required columns: {missing}")

    classifier_llm_config = {"model": args.classifier_model, "temperature": 0.0}
    groups = CUE_GROUP_CONFIGS[args.config]

    print(f"Running baseline (13 singleton calls/turn) on {len(df)} turns...")
    baseline = run_baseline(df, classifier_llm_config)

    print(f"Running grouped config '{args.config}' ({len(groups)} calls/turn) on {len(df)} turns...")
    grouped = run_grouped(df, classifier_llm_config, groups)

    report = compare(baseline, grouped)
    report.to_csv(args.out, index=False)
    print(report.to_string(index=False))
    print(f"\nOverall mean raw agreement: {report['raw_agreement'].mean():.3f}")
    reduction = 100 * (1 - len(groups) / len(ALL_LLM_CUES))
    print(f"Calls/turn: baseline=13, grouped={len(groups)} ({reduction:.0f}% fewer calls)")
    print(f"Full comparison saved to: {args.out}")


if __name__ == "__main__":
    main()
