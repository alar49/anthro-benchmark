# Mistral Small 4 checks (analysis plan, Section 11). Descriptive only.
#
#   Check 1, subsampling: does the 96-scenario seed42 subsample represent the
#     full scenario set? seed42_only vs. the other 864 dialogues of the SAME
#     run (all960). The 96 are part of the 960, so comparing them with all 960
#     would compare a group with itself plus others: the difference is diluted
#     (subsample - all960 = 0.9 x (subsample - rest)) and the two estimates are
#     not independent. Comparing with the other 864 asks the same question
#     with valid intervals.
#   Check 2, reproducibility: does a second run on the same 96 scenarios give
#     the same results? mistral_small_4_test vs. seed42_only, paired by
#     scenario. Only meaningful if the test run was rated with the final
#     classifier.
#
# Per dialogue it computes, for the 13-cue index and each cue: the rate over
# all turns, the rate in turns 1..split_at, and the late-minus-early gap; plus
# the dialogue's length and whether it reached max_turns. Groups are compared
# on the average over dialogues (each dialogue counts once).
#
# Equivalence: a difference counts as negligible only if its 90% interval lies
# within +/- the margin (the interval form of two one-sided tests at alpha =
# .05). A 95% interval that includes zero is not enough: absence of evidence
# is not evidence of equivalence.
#
# Usage:
#   python subsample_checks.py \
#       --full mistral_small_4_all960_full=<all960 rated CSV> \
#       --subsample mistral_small_4_seed42_only=<seed42 rated CSV> \
#       --test mistral_small_4_test=<test rated CSV> \
#       --scenario-col <column identifying the scenario> \
#       --overrides status_overrides.csv \
#       --output-dir subsample_checks_results
#
# Without --scenario-col, check 1 splits the 960 by dialogue_id (seed42_only
# was extracted from all960, so its ids should match) and check 2 is skipped,
# because pairing two runs needs a scenario key.

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import turn_dynamics as td

PCT_STATS = ("rate_all_pct", "rate_early_pct", "gap_pp", "reached_full_pct")


def dialogue_values(
    run_df: pd.DataFrame,
    cue_cols: List[str],
    index_cols: List[str],
    scenario_col: Optional[str],
    split_at: int,
    max_turns: int,
) -> pd.DataFrame:
    """One row per dialogue x measure x statistic."""
    d = td.prepare(run_df, cue_cols, split_at, max_turns)
    ml = td.build_measure_long(td.build_messages(d, index_cols), td.build_cue_long(d, cue_cols))
    keys = d.groupby("dkey").agg(
        dialogue_id=("dialogue_id", "first"),
        length=("length", "first"),
        n_early_rated=("n_early_rated", "first"),
        n_late_rated=("n_late_rated", "first"),
    )
    if scenario_col:
        keys["scenario"] = d.groupby("dkey")[scenario_col].first().astype(str)

    sums = ml.groupby(["dkey", "measure", "window"])[["present", "valid"]].sum().unstack("window", fill_value=0)
    sums.columns = [f"{w}_{v}" for v, w in sums.columns]
    for c in ("early_present", "early_valid", "late_present", "late_valid"):
        if c not in sums.columns:
            sums[c] = 0
    s = sums.reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        all_valid = s["early_valid"] + s["late_valid"]
        s["rate_all_pct"] = np.where(all_valid > 0, (s["early_present"] + s["late_present"]) / all_valid * 100, np.nan)
        s["rate_early_pct"] = np.where(s["early_valid"] > 0, s["early_present"] / s["early_valid"] * 100, np.nan)
        late = np.where(s["late_valid"] > 0, s["late_present"] / s["late_valid"] * 100, np.nan)
    s = s.merge(keys[["n_early_rated", "n_late_rated"]], left_on="dkey", right_index=True)
    # same dialogues as Section 4: all early turns rated, at least one rated late turn
    ok_gap = (s["n_early_rated"] == split_at) & (s["n_late_rated"] >= 1)
    s["gap_pp"] = np.where(ok_gap, late - s["rate_early_pct"], np.nan)
    long = s.melt(
        id_vars=["dkey", "measure"], value_vars=["rate_all_pct", "rate_early_pct", "gap_pp"],
        var_name="statistic", value_name="value",
    )
    dl = keys.reset_index()
    dl_long = pd.concat(
        [
            dl[["dkey"]].assign(measure="dialogue", statistic="length_turns", value=dl["length"]),
            dl[["dkey"]].assign(measure="dialogue", statistic="reached_full_pct",
                                value=(dl["length"] == max_turns).astype(float) * 100),
        ],
        ignore_index=True,
    )
    out = pd.concat([long, dl_long], ignore_index=True).dropna(subset=["value"])
    cols = ["dkey", "dialogue_id"] + (["scenario"] if scenario_col else [])
    return out.merge(keys.reset_index()[cols], on="dkey")


def _intervals(boot: np.ndarray) -> Tuple[float, float, float, float]:
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return (np.nan,) * 4
    lo95, hi95, lo90, hi90 = np.percentile(boot, [2.5, 97.5, 5, 95])
    return float(lo95), float(hi95), float(lo90), float(hi90)


def _margin(statistic: str, margin_pp: float, margin_turns: float) -> float:
    return margin_pp if statistic in PCT_STATS else margin_turns


def compare_independent(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, rng,
                        margin_pp: float, margin_turns: float) -> pd.DataFrame:
    """Check 1: mean of group a minus mean of group b, resampling dialogues
    within each group."""
    rows = []
    for (measure, stat), ga in a.groupby(["measure", "statistic"], sort=True):
        gb = b[(b["measure"] == measure) & (b["statistic"] == stat)]
        va, vb = ga["value"].to_numpy(float), gb["value"].to_numpy(float)
        if len(va) < 2 or len(vb) < 2:
            continue
        diff = va.mean() - vb.mean()
        ia = rng.integers(0, len(va), size=(n_boot, len(va)))
        ib = rng.integers(0, len(vb), size=(n_boot, len(vb)))
        boot = va[ia].mean(axis=1) - vb[ib].mean(axis=1)
        lo95, hi95, lo90, hi90 = _intervals(boot)
        m = _margin(stat, margin_pp, margin_turns)
        rows.append(dict(
            measure=measure, statistic=stat, n_subsample=len(va), n_rest=len(vb),
            mean_subsample=round(va.mean(), 2), mean_rest=round(vb.mean(), 2), diff=round(diff, 2),
            ci95_low=round(lo95, 2), ci95_high=round(hi95, 2), ci90_low=round(lo90, 2), ci90_high=round(hi90, 2),
            margin=m, equivalent_within_margin=bool(lo90 > -m and hi90 < m),
        ))
    return pd.DataFrame(rows)


def compare_paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, rng,
                   margin_pp: float, margin_turns: float) -> pd.DataFrame:
    """Check 2: mean paired difference a - b over scenarios present in both,
    resampling scenarios; Spearman correlation across scenarios."""
    rows = []
    for (measure, stat), ga in a.groupby(["measure", "statistic"], sort=True):
        gb = b[(b["measure"] == measure) & (b["statistic"] == stat)]
        pair = ga[["scenario", "value"]].merge(gb[["scenario", "value"]], on="scenario", suffixes=("_test", "_seed42"))
        if len(pair) < 2:
            continue
        d = (pair["value_test"] - pair["value_seed42"]).to_numpy(float)
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        lo95, hi95, lo90, hi90 = _intervals(d[idx].mean(axis=1))
        rho = pair["value_test"].rank().corr(pair["value_seed42"].rank()) if len(pair) >= 3 else np.nan
        m = _margin(stat, margin_pp, margin_turns)
        rows.append(dict(
            measure=measure, statistic=stat, n_scenarios=len(pair),
            mean_test=round(pair["value_test"].mean(), 2), mean_seed42=round(pair["value_seed42"].mean(), 2),
            mean_paired_diff=round(d.mean(), 2), ci95_low=round(lo95, 2), ci95_high=round(hi95, 2),
            ci90_low=round(lo90, 2), ci90_high=round(hi90, 2), margin=m,
            equivalent_within_margin=bool(lo90 > -m and hi90 < m),
            spearman_across_scenarios=round(float(rho), 3) if not np.isnan(rho) else np.nan,
        ))
    return pd.DataFrame(rows)


def run(
    full: Tuple[str, str],
    subsample: Tuple[str, str],
    test: Optional[Tuple[str, str]],
    output_dir: str = "subsample_checks_results",
    scenario_col: Optional[str] = None,
    overrides: Optional[str] = None,
    pronoun_cue: str = "personal pronoun use",
    split_at: int = 5,
    max_turns: int = 10,
    keep_failed: bool = False,
    keep_after_empty_reply: bool = False,
    margin_pp: float = 5.0,
    margin_turns: float = 0.5,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    specs = [full, subsample] + ([test] if test else [])
    raw = td.load_runs(specs)
    if scenario_col and scenario_col not in raw.columns:
        raise ValueError(f"--scenario-col '{scenario_col}' is not a column of the rated CSVs.")
    cue_cols = td.find_cue_columns(raw)
    raw, log, applied = td.apply_overrides(raw, overrides)
    if not keep_failed:
        failed = raw["dialogue_status"].map(td.status_bucket) == "failed"
        log.append("Failed dialogues excluded per run: "
                   + str(raw[failed].groupby("model")["dialogue_id"].nunique().to_dict()))
    else:
        failed = pd.Series(False, index=raw.index)
    names = [td.cue_name(c) for c in cue_cols]
    index_cols = [f"{n}_present" for n in names if n != pronoun_cue]
    full_label, sub_label = full[0], subsample[0]
    split_key = scenario_col or "dialogue_id"
    sub_keys = set(raw.loc[raw["model"] == sub_label, split_key].astype(str))  # before any cut

    # Same rule as the main analysis: cut each dialogue at its first empty reply
    empties = td.empty_reply_table(raw, cue_cols)
    if not keep_after_empty_reply:
        raw = td.cut_at_empty_reply(raw, empties)
    log.append("Dialogues with an empty reply per run: " + str(empties.groupby("model").size().to_dict())
               + (" (cut at the first one)" if not keep_after_empty_reply else " (kept, sensitivity run)"))

    # Check 1 split: the rest of the full run = dialogues whose scenario (or id) is not in the subsample run
    sub_raw = raw[raw["model"] == sub_label]
    full_raw = raw[raw["model"] == full_label]
    in_sub = full_raw[split_key].astype(str).isin(sub_keys)
    found = full_raw.loc[in_sub, split_key].astype(str).nunique()
    log.append(f"Check 1 split by '{split_key}': {found} of {len(sub_keys)} subsample keys found in the full run.")
    if found < len(sub_keys):
        log.append("WARNING: not every subsample key is in the full run; check --scenario-col.")
    rest_raw = full_raw[~in_sub].assign(model=f"{full_label}_rest")

    def values(frame, label):
        frame = frame[~failed.reindex(frame.index, fill_value=False)].assign(model=label)
        return dialogue_values(frame, cue_cols, index_cols, scenario_col, split_at, max_turns)

    v_sub = values(sub_raw, sub_label)
    v_rest = values(rest_raw, f"{full_label}_rest")
    check1 = compare_independent(v_sub, v_rest, n_boot, rng, margin_pp, margin_turns)
    outputs = {"subsample_vs_rest.csv": check1}
    frames = [v_sub.assign(group=sub_label), v_rest.assign(group=f"{full_label}_rest")]

    if test and scenario_col:
        v_test = values(raw[raw["model"] == test[0]], test[0])
        check2 = compare_paired(v_test, v_sub, n_boot, rng, margin_pp, margin_turns)
        outputs["test_vs_seed42_paired.csv"] = check2
        frames.append(v_test.assign(group=test[0]))
        n_pairs = len(set(v_test["scenario"]) & set(v_sub["scenario"]))
        log.append(f"Check 2: {n_pairs} scenarios present in both runs.")
    elif test:
        log.append("Check 2 skipped: pairing the two runs needs --scenario-col.")

    outputs["dialogue_values.csv"] = pd.concat(frames, ignore_index=True)
    outputs["status_overrides_applied.csv"] = applied
    outputs["empty_replies.csv"] = empties
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(output_dir, name), index=False)
    with open(os.path.join(output_dir, "checks_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([f"runs: {specs}", f"scenario_col: {scenario_col}", f"margin_pp: {margin_pp}",
                           f"margin_turns: {margin_turns}", f"n_boot: {n_boot}", f"seed: {seed}", ""] + log) + "\n")
    print("\n".join(log))
    print(f"Results in {os.path.abspath(output_dir)}")
    return {k.replace(".csv", ""): v for k, v in outputs.items()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mistral Small 4 subsampling and reproducibility checks.")
    p.add_argument("--full", required=True, help="label=path to the all960 rated CSV")
    p.add_argument("--subsample", required=True, help="label=path to the seed42_only rated CSV")
    p.add_argument("--test", default=None, help="label=path to the test-run rated CSV (check 2)")
    p.add_argument("--scenario-col", default=None, help="Column identifying the scenario (needed for check 2)")
    p.add_argument("--overrides", default=None)
    p.add_argument("--output-dir", default="subsample_checks_results")
    p.add_argument("--pronoun-cue", default="personal pronoun use")
    p.add_argument("--split-at", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--keep-failed", action="store_true")
    p.add_argument("--keep-after-empty-reply", action="store_true")
    p.add_argument("--margin-pp", type=float, default=5.0, help="Equivalence margin for percentages")
    p.add_argument("--margin-turns", type=float, default=0.5, help="Equivalence margin for dialogue length")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    one = lambda s: td.parse_run_specs([s])[0]  # noqa: E731
    run(one(a.full), one(a.subsample), one(a.test) if a.test else None,
        output_dir=a.output_dir, scenario_col=a.scenario_col, overrides=a.overrides,
        pronoun_cue=a.pronoun_cue, split_at=a.split_at, max_turns=a.max_turns,
        keep_failed=a.keep_failed, keep_after_empty_reply=a.keep_after_empty_reply, margin_pp=a.margin_pp, margin_turns=a.margin_turns,
        n_boot=a.n_boot, seed=a.seed)
