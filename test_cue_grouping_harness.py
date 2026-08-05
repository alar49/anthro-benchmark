"""
Compares the current one-cue-per-call baseline (classifiers.LLMClassifier,
as used in rating.py's rate_dialogues loop) against a candidate cue grouping
(cue_grouping.LLMGroupClassifier), on the same pilot dialogue set, and
reports per-cue label agreement so you can decide whether a grouping is
safe to promote to production before running it on the full dataset.

Usage:
    python test_cue_grouping_harness.py --dialogues pilot_dialogues.csv \
        --config A_interference_avoidant --classifier-model <model_name> \
        [--classifier-max-tokens-base N] [--classifier-max-tokens-per-cue N]

pilot_dialogues.csv needs at least user_message and assistant_message
columns (same schema rate_dialogues() expects).

--classifier-max-tokens-base/--classifier-max-tokens-per-cue mirror
rate_dialogues()'s flags exactly (same formula, computed per call unit --
1 cue for every baseline call, len(group) for each grouped call), so you
can pilot a token budget here before trusting it on a full run, rather
than only being able to test it in production.

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


def _config_with_max_tokens(
    base_config: dict, num_cues: int, max_tokens_base: int | None, max_tokens_per_cue: int | None
) -> dict:
    """Same formula as rating.py's rate_dialogues(): base + per_cue * (cues
    in this call). Returns a copy -- base_config is shared across many
    classifiers with different num_cues, so it must not be mutated."""
    config = dict(base_config)
    if max_tokens_base or max_tokens_per_cue:
        max_tokens = (max_tokens_base or 0) + (max_tokens_per_cue or 0) * num_cues
        if max_tokens > 0:
            config["max_tokens"] = max_tokens
    return config


def _classify_error(reason: str) -> str:
    """Bucket a -1 rating's reason string by why it failed, using the
    diagnostic prefixes _parse_grouped_output already writes. Baseline
    (singleton) failures won't match any of these -- LLMClassifier
    returns the model's raw text verbatim rather than a synthesized
    diagnostic -- so they correctly fall into 'other_or_unformatted'
    rather than being misclassified as something more specific."""
    if "no line found" in reason and ("truncated" in reason or "malformed" in reason):
        return "missing_or_truncated"
    if "ambiguous label" in reason:
        return "ambiguous_label"
    if "could not parse JSON" in reason:
        return "unparseable_json"
    return "other_or_unformatted"


def run_baseline(
    df: pd.DataFrame,
    classifier_llm_config: dict,
    max_tokens_base: int | None = None,
    max_tokens_per_cue: int | None = None,
) -> dict:
    """One LLMClassifier per cue, called once per turn -- exactly what
    rate_dialogues() does today. 13 calls/turn. Every baseline call unit
    is 1 cue, so max_tokens (if set) is max_tokens_base + max_tokens_per_cue."""
    results = {cue: [] for cue in ALL_LLM_CUES}
    classifiers = {
        cue: LLMClassifier(
            _config_with_max_tokens(classifier_llm_config, 1, max_tokens_base, max_tokens_per_cue),
            cue,
            CUE_DEFINITIONS[cue]["definition"],
            CUE_DEFINITIONS[cue].get("examples", []),
        )
        for cue in ALL_LLM_CUES
    }
    for _, row in df.iterrows():
        for cue in ALL_LLM_CUES:
            score, reason = classifiers[cue].rate_turn_messages(
                cue=cue,
                assistant_turn_message=row["assistant_message"],
                user_turn_message=row["user_message"],
            )
            results[cue].append((score, reason))
    return results


def run_grouped(
    df: pd.DataFrame,
    classifier_llm_config: dict,
    groups: list,
    max_tokens_base: int | None = None,
    max_tokens_per_cue: int | None = None,
) -> dict:
    """One LLMGroupClassifier per group, called once per turn. len(groups)
    calls/turn. Each group's max_tokens (if set) is scaled by its own
    size, same formula rate_dialogues() uses per call unit."""
    results = {cue: [] for group in groups for cue in group}
    group_classifiers = [
        LLMGroupClassifier(
            _config_with_max_tokens(classifier_llm_config, len(group), max_tokens_base, max_tokens_per_cue),
            group,
        )
        for group in groups
    ]
    for _, row in df.iterrows():
        for gc in group_classifiers:
            scores = gc.rate_turn_messages(row["assistant_message"], row["user_message"])
            for cue, (score, reason) in scores.items():
                results[cue].append((score, reason))
    return results


def compare(baseline: dict, grouped: dict) -> pd.DataFrame:
    rows = []
    for cue in ALL_LLM_CUES:
        b, g = baseline[cue], grouped[cue]
        b_scores, g_scores = [s for s, _ in b], [s for s, _ in g]
        n = len(b_scores)
        agree = sum(1 for x, y in zip(b_scores, g_scores) if x == y)

        b_error_reasons = [r for s, r in b if s == -1]
        g_error_reasons = [r for s, r in g if s == -1]
        b_error_types = {}
        g_error_types = {}
        for r in b_error_reasons:
            kind = _classify_error(r)
            b_error_types[kind] = b_error_types.get(kind, 0) + 1
        for r in g_error_reasons:
            kind = _classify_error(r)
            g_error_types[kind] = g_error_types.get(kind, 0) + 1

        rows.append(
            {
                "cue": cue,
                "n": n,
                "raw_agreement": agree / n if n else float("nan"),
                "baseline_yes_rate": (sum(1 for x in b_scores if x == 1) / n) if n else float("nan"),
                "grouped_yes_rate": (sum(1 for x in g_scores if x == 1) / n) if n else float("nan"),
                "baseline_parse_errors": len(b_error_reasons),
                "grouped_parse_errors": len(g_error_reasons),
                "grouped_errors_missing_or_truncated": g_error_types.get("missing_or_truncated", 0),
                "grouped_errors_ambiguous_label": g_error_types.get("ambiguous_label", 0),
                "grouped_errors_other": g_error_types.get("unparseable_json", 0)
                + g_error_types.get("other_or_unformatted", 0),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", required=True, help="Pilot dialogues CSV (user_message, assistant_message columns)")
    parser.add_argument("--config", required=True, choices=list(CUE_GROUP_CONFIGS.keys()))
    parser.add_argument("--classifier-model", required=True)
    parser.add_argument(
        "--classifier-max-tokens-base",
        type=int,
        default=None,
        help="See rate_dialogues()'s flag of the same name -- same formula, applied here per call unit so you can pilot a token budget before trusting it on a full run.",
    )
    parser.add_argument(
        "--classifier-max-tokens-per-cue",
        type=int,
        default=None,
        help="See rate_dialogues()'s flag of the same name.",
    )
    parser.add_argument("--out", default="cue_grouping_comparison.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.dialogues)
    missing = {"user_message", "assistant_message"} - set(df.columns)
    if missing:
        raise ValueError(f"dialogues CSV missing required columns: {missing}")

    classifier_llm_config = {"model": args.classifier_model, "temperature": 0.0}
    groups = CUE_GROUP_CONFIGS[args.config]

    print(f"Running baseline (13 singleton calls/turn) on {len(df)} turns...")
    baseline = run_baseline(
        df, classifier_llm_config, args.classifier_max_tokens_base, args.classifier_max_tokens_per_cue
    )

    print(f"Running grouped config '{args.config}' ({len(groups)} calls/turn) on {len(df)} turns...")
    grouped = run_grouped(
        df, classifier_llm_config, groups, args.classifier_max_tokens_base, args.classifier_max_tokens_per_cue
    )

    report = compare(baseline, grouped)
    report.to_csv(args.out, index=False)
    print(report.to_string(index=False))
    print(f"\nOverall mean raw agreement: {report['raw_agreement'].mean():.3f}")
    reduction = 100 * (1 - len(groups) / len(ALL_LLM_CUES))
    print(f"Calls/turn: baseline=13, grouped={len(groups)} ({reduction:.0f}% fewer calls)")
    if args.classifier_max_tokens_base or args.classifier_max_tokens_per_cue:
        print(
            f"max_tokens: base={args.classifier_max_tokens_base or 0}, "
            f"per_cue={args.classifier_max_tokens_per_cue or 0} "
            f"(baseline calls: {(args.classifier_max_tokens_base or 0) + (args.classifier_max_tokens_per_cue or 0)}; "
            f"grouped config's largest unit ({max(len(g) for g in groups)} cues): "
            f"{(args.classifier_max_tokens_base or 0) + (args.classifier_max_tokens_per_cue or 0) * max(len(g) for g in groups)})"
        )
        if (report["grouped_errors_missing_or_truncated"] > 0).any():
            print(
                "Note: some grouped-call errors are classified as missing_or_truncated -- "
                "check whether max_tokens is too tight for this config before blaming interference."
            )
    print(f"Full comparison saved to: {args.out}")


if __name__ == "__main__":
    main()
