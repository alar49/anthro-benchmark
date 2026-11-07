# AnthroBench turn-dynamics analysis -- Python part.
#
# Implements the data rules and the descriptive analyses of the analysis plan
# ("AnthroBench turn analysis -- analysis plan"), and writes the inputs for the
# mixed models in turn_dynamics_models.R:
#
#   Section 2  inclusion: status overrides, failed-dialogue exclusion,
#              95% completion rule, rated turns, windows, cue sets
#   Section 4  how much a 5-turn benchmark misses (same dialogues, bootstrap)
#   Section 6  emergence of new cues (first appearances, per-turn hazard)
#   Section 7  persistence, descriptive companion
#   Section 8  personal pronoun use and the between-model table
#   R inputs   messages.csv (Section 3), cue_long.csv (Sections 5 and 7),
#              emergence_risk.csv (Section 6, first-appearance model)
#
# Why everything is computed per dialogue and resampled by dialogue:
#   messages from the same dialogue share the scenario, the persona and the
#   conversation history, so they resemble each other. 1,000 messages from
#   100 dialogues carry less information than 1,000 independent messages
#   (with 10 messages per dialogue and an intraclass correlation of 0.3,
#   about as much as 270). Treating messages as independent would make every
#   interval too narrow. The bootstrap below therefore resamples whole
#   dialogues, and the R models give each dialogue its own baseline.
#
# Usage:
#   python turn_dynamics.py \
#       --run deepseek_v4_flash=results/deepseek_v4_flash/rated.csv \
#       --run gpt_oss_120b=results/gpt_oss_120b/rated.csv \
#       --overrides status_overrides.csv \
#       --model-info model_info.csv \
#       --output-dir turn_dynamics_results
#   Rscript turn_dynamics_models.R turn_dynamics_results
#
# status_overrides.csv: model, dialogue_id, status
#   (e.g. mistral_nemo, 1234, completed_early_natural_end*)
# model_info.csv: model, release_date (YYYY-MM-DD), params_total_b,
#   params_active_b, family. params_active_b = active parameters for
#   mixture-of-experts models, effective size for e2b/e4b models, total
#   otherwise; leave blank when there is no public count.

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FAILED_PREFIX = "failed_at_turn_"
BUDGET_PREFIX = "stopped_budget_exceeded_"
INDEX = "index_13"  # name of the 13-cue index measure in the outputs


# ---------------------------------------------------------------------------
# Loading and inclusion (plan, Section 2)
# ---------------------------------------------------------------------------


def read_csv_strict(path: str, **kwargs) -> pd.DataFrame:
    """pd.read_csv with a clear message when a file does not follow the
    standard format (comma-separated, UTF-8): see check_csv_format.py."""
    hint = f"Check and normalize it with: python check_csv_format.py \"{path}\" --fix-dir normalized"
    try:
        df = pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        raise ValueError(f"{path} is not UTF-8 encoded. {hint}") from None
    if df.shape[1] == 1 and any(s in str(df.columns[0]) for s in (";", "\t")):
        raise ValueError(f"{path} is not comma-separated. {hint}")
    return df


def parse_run_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """'label=path' -> (label, path)."""
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--run expects label=path, got: {s}")
        label, path = s.split("=", 1)
        out.append((label.strip(), path.strip()))
    labels = [l for l, _ in out]
    if len(set(labels)) != len(labels):
        raise ValueError("Each --run label must be unique.")
    return out


def load_runs(run_specs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for label, path in run_specs:
        df = read_csv_strict(path)
        for col in ("dialogue_id", "turn_pair_index"):
            if col not in df.columns:
                raise ValueError(f"{path}: missing column '{col}'.")
        df["model"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def find_cue_columns(df: pd.DataFrame) -> List[str]:
    """Same selection rule as load_data() in the analysis module."""
    cols = [
        c
        for c in df.columns
        if c.endswith("_present") and "_s" not in c and "_final_present" not in c
    ]
    if not cols:
        raise ValueError("No '*_present' cue columns found.")
    return cols


def cue_name(col: str) -> str:
    return col[: -len("_present")]


def apply_overrides(
    df: pd.DataFrame, path: Optional[str]
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Replaces dialogue_status for the (model, dialogue_id) pairs listed in the
    overrides file. The original status is kept in status_original, and the
    third return value lists every relabeled dialogue with both values."""
    notes: List[str] = []
    applied = pd.DataFrame(columns=["model", "dialogue_id", "status_original", "status_new"])
    if "dialogue_status" not in df.columns:
        df["dialogue_status"] = np.nan
        notes.append("No dialogue_status column: completion rule and failed-dialogue exclusion cannot be applied.")
    df["dialogue_status"] = df["dialogue_status"].astype(object)
    df["status_original"] = df["dialogue_status"]
    if not path:
        return df, notes, applied
    ov = read_csv_strict(path)
    missing = {"model", "dialogue_id", "status"} - set(ov.columns)
    if missing:
        raise ValueError(f"Overrides file needs columns model, dialogue_id, status; missing {missing}.")
    ov = ov.assign(_k=ov["model"].astype(str) + "::" + ov["dialogue_id"].astype(str))
    key = df["model"].astype(str) + "::" + df["dialogue_id"].astype(str)
    mapping = dict(zip(ov["_k"], ov["status"]))
    hit = key.isin(mapping.keys())
    df.loc[hit, "dialogue_status"] = key[hit].map(mapping)
    unmatched = sorted(set(mapping) - set(key[hit]))
    notes.append(f"Status overrides applied to {key[hit].nunique()} dialogues.")
    if unmatched:
        notes.append(f"Overrides with no matching dialogue (check model labels and ids): {unmatched}")
    applied = (
        df.loc[hit, ["model", "dialogue_id", "status_original", "dialogue_status"]]
        .drop_duplicates(subset=["model", "dialogue_id"])
        .rename(columns={"dialogue_status": "status_new"})
        .reset_index(drop=True)
    )
    return df, notes, applied


def status_bucket(status) -> str:
    if pd.isna(status):
        return "unknown"
    s = str(status)
    if s.startswith(FAILED_PREFIX):
        return "failed"
    if s.startswith(BUDGET_PREFIX):
        return "budget_capped"
    if s.startswith("completed"):  # includes relabeled values such as "completed_early_natural_end*"
        return "completed"
    return "unknown"


def completion_report(df: pd.DataFrame) -> pd.DataFrame:
    per_dialogue = df.groupby(["model", "dialogue_id"], sort=False)["dialogue_status"].first()
    buckets = per_dialogue.map(status_bucket).rename("bucket").reset_index()
    tab = pd.crosstab(buckets["model"], buckets["bucket"])
    for c in ("completed", "failed", "budget_capped", "unknown"):
        if c not in tab.columns:
            tab[c] = 0
    tab["dialogues"] = tab[["completed", "failed", "budget_capped", "unknown"]].sum(axis=1)
    known = tab["dialogues"] - tab["unknown"]
    tab["completion_rate"] = np.where(known > 0, tab["completed"] / tab["dialogues"], np.nan)
    return tab[["dialogues", "completed", "failed", "budget_capped", "unknown", "completion_rate"]].reset_index()


def apply_inclusion(
    df: pd.DataFrame,
    report: pd.DataFrame,
    min_completion: Optional[float],
    keep_failed: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    notes = []
    report = report.copy()
    report["included"] = True
    if min_completion is not None:
        low = report["completion_rate"] < min_completion  # NaN -> False: rule not applicable
        report.loc[low, "included"] = False
        if low.any():
            notes.append(
                f"Models below the {min_completion:.0%} completion rule, excluded: "
                f"{report.loc[low, 'model'].tolist()}"
            )
    df = df[df["model"].isin(report.loc[report["included"], "model"])]
    if not keep_failed:
        failed = df["dialogue_status"].map(status_bucket) == "failed"
        n_failed = df.loc[failed, ["model", "dialogue_id"]].drop_duplicates().shape[0]
        df = df[~failed]
        notes.append(f"Failed dialogues excluded: {n_failed}.")
    else:
        notes.append("Failed dialogues kept up to the turn where they failed (sensitivity run).")
    return df.copy(), report, notes


# ---------------------------------------------------------------------------
# Preparation: rated turns, windows, measures
# ---------------------------------------------------------------------------


def prepare(
    df: pd.DataFrame, cue_cols: Sequence[str], split_at: int, max_turns: int
) -> pd.DataFrame:
    """Keeps rated turns only (at least one cue with a 0/1 rating) within the
    first max_turns positions, and adds dkey, turn (1-based), window, length."""
    d = df.copy()
    d["dkey"] = d["model"].astype(str) + "::" + d["dialogue_id"].astype(str)
    d["turn"] = d["turn_pair_index"].astype(int) + 1
    rated = d[list(cue_cols)].isin([0, 1]).any(axis=1)
    d = d[rated & (d["turn"] <= max_turns)].copy()
    d["window"] = np.where(d["turn"] <= split_at, "early", "late")
    d["_is_early"] = (d["window"] == "early").astype(int)
    g = d.groupby("dkey")
    d["length"] = g["turn"].transform("size")
    d["n_early_rated"] = g["_is_early"].transform("sum")
    d["n_late_rated"] = d["length"] - d["n_early_rated"]
    return d.drop(columns="_is_early")


def build_messages(d: pd.DataFrame, index_cols: Sequence[str]) -> pd.DataFrame:
    """One row per rated message: k of n index cues present (input for Section 3)."""
    vals = d[list(index_cols)]
    out = d[["model", "dkey", "turn", "window", "length", "n_early_rated", "n_late_rated"]].copy()
    out["k"] = vals.eq(1).sum(axis=1).astype(int)
    out["n"] = vals.isin([0, 1]).sum(axis=1).astype(int)
    return out[out["n"] > 0].reset_index(drop=True)


def build_cue_long(d: pd.DataFrame, cue_cols: Sequence[str]) -> pd.DataFrame:
    """One row per (message, cue) with a valid 0/1 rating, plus the previous
    turn's rating (prev, only if turn t-1 exists and is validly rated) and the
    turn-1 rating (first). Input for Sections 5 and 7."""
    base = ["model", "dkey", "turn", "window", "length", "n_early_rated", "n_late_rated"]
    long = d[base + list(cue_cols)].melt(
        id_vars=base, value_vars=list(cue_cols), var_name="cue", value_name="value"
    )
    long["cue"] = long["cue"].map(cue_name)
    long = long.sort_values(["dkey", "cue", "turn"]).reset_index(drop=True)
    g = long.groupby(["dkey", "cue"], sort=False)
    prev_value = g["value"].shift(1)
    prev_turn = g["turn"].shift(1)
    ok_prev = (prev_turn == long["turn"] - 1) & prev_value.isin([0, 1])
    long["prev"] = np.where(ok_prev, prev_value, np.nan)
    first = long[(long["turn"] == 1) & long["value"].isin([0, 1])].set_index(["dkey", "cue"])["value"]
    long["first"] = pd.MultiIndex.from_frame(long[["dkey", "cue"]]).map(first.to_dict().get)
    long["first"] = pd.to_numeric(long["first"], errors="coerce")
    long = long[long["value"].isin([0, 1])].copy()
    long["present"] = long["value"].astype(int)
    return long.drop(columns="value").reset_index(drop=True)


def build_measure_long(messages: pd.DataFrame, cue_long: pd.DataFrame) -> pd.DataFrame:
    """Unified table: one row per message and measure (the 13-cue index or a
    single cue), with present / valid counts."""
    cols = ["model", "dkey", "turn", "window", "length", "n_early_rated", "n_late_rated"]
    idx = messages[cols].assign(measure=INDEX, present=messages["k"], valid=messages["n"])
    cue = cue_long[cols].assign(measure=cue_long["cue"], present=cue_long["present"], valid=1)
    return pd.concat([idx, cue], ignore_index=True)


# ---------------------------------------------------------------------------
# Dialogue-level bootstrap
# ---------------------------------------------------------------------------


def _pct_ci(values: np.ndarray) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def boot_ratio(num, den, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    """sum(num)/sum(den) with a percentile CI, resampling dialogues (rows)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    point = num.sum() / den.sum() if den.sum() > 0 else np.nan
    if len(num) < 2 or n_boot <= 0:
        return point, np.nan, np.nan
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = _pct_ci(r)
    return point, lo, hi


def boot_ratio_diff(num_a, den_a, num_b, den_b, n_boot: int, rng: np.random.Generator):
    """(sum(num_b)/sum(den_b)) - (sum(num_a)/sum(den_a)), same resampled
    dialogues for both ratios (paired)."""
    a = [np.asarray(x, float) for x in (num_a, den_a, num_b, den_b)]
    na, da, nb, db = a
    with np.errstate(invalid="ignore", divide="ignore"):
        point = nb.sum() / db.sum() - na.sum() / da.sum()
    if len(na) < 2 or n_boot <= 0:
        return point, np.nan, np.nan
    idx = rng.integers(0, len(na), size=(n_boot, len(na)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = nb[idx].sum(1) / db[idx].sum(1) - na[idx].sum(1) / da[idx].sum(1)
    lo, hi = _pct_ci(r)
    return float(point), lo, hi


def _r(x, digits=2):
    return round(float(x), digits) if x is not None and not np.isnan(x) else np.nan


# ---------------------------------------------------------------------------
# Section 4: how much a 5-turn benchmark misses
# ---------------------------------------------------------------------------


def five_vs_ten(ml: pd.DataFrame, split_at: int, n_boot: int, rng) -> pd.DataFrame:
    """Dialogues with all early turns rated and at least one rated late turn;
    pooled rate in turns 1..split_at vs. the rest, same dialogues."""
    cohort = ml[(ml["n_early_rated"] == split_at) & (ml["n_late_rated"] >= 1)]
    sums = cohort.groupby(["model", "measure", "dkey", "window"])[["present", "valid"]].sum()
    wide = sums.unstack("window", fill_value=0)
    wide.columns = [f"{w}_{v}" for v, w in wide.columns]
    for c in ("early_present", "early_valid", "late_present", "late_valid"):
        if c not in wide.columns:
            wide[c] = 0
    wide = wide.reset_index()
    # Each dialogue counts once in each window: rate per dialogue first, then
    # the average over dialogues. Pooling messages instead would give the
    # dialogues with more late turns -- the ones that ran longer, which may be
    # "warmer" -- more weight in the late window than in the early one, and
    # recreate the selection effect inside the same set of dialogues.
    rows = []
    for (model, measure), g in wide.groupby(["model", "measure"], sort=True):
        g = g[(g["early_valid"] > 0) & (g["late_valid"] > 0)]
        if g.empty:
            continue
        early = (g["early_present"] / g["early_valid"]).to_numpy()
        late = (g["late_present"] / g["late_valid"]).to_numpy()
        diff, lo, hi = boot_ratio(late - early, np.ones(len(g)), n_boot, rng)
        rows.append(
            dict(
                model=model,
                measure=measure,
                n_dialogues=len(g),
                early_pct=_r(early.mean() * 100),
                late_pct=_r(late.mean() * 100),
                diff_pp=_r(diff * 100),
                ci95_low_pp=_r(lo * 100),
                ci95_high_pp=_r(hi * 100),
            )
        )
    return pd.DataFrame(rows)


def rates_by_turn_and_length(ml: pd.DataFrame, length_bins: Sequence[int]) -> pd.DataFrame:
    """Rate at each turn, separately per dialogue-length group (assumption check
    for Section 3: rising lines = within-dialogue increase, stacked flat lines =
    selection)."""
    edges = list(length_bins)
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo + 1}-{hi}" if hi - lo > 1 else f"{hi}")
    labels[0] = f"<={edges[1]}"
    grp = pd.cut(ml["length"], bins=edges, labels=labels, include_lowest=True)
    out = (
        ml.assign(length_group=grp)
        .groupby(["model", "measure", "length_group", "turn"], observed=True)
        .agg(n_dialogues=("dkey", "nunique"), present=("present", "sum"), valid=("valid", "sum"))
        .reset_index()
    )
    out["pct"] = (out["present"] / out["valid"] * 100).round(2)
    return out


# ---------------------------------------------------------------------------
# Section 6: emergence of new cues
# ---------------------------------------------------------------------------


def first_seen(cue_long: pd.DataFrame) -> pd.Series:
    """Turn at which each (dialogue, cue) is first present; NaN if never."""
    return cue_long[cue_long["present"] == 1].groupby(["dkey", "cue"])["turn"].min()


def emergence(
    cue_long: pd.DataFrame,
    index_cues: Sequence[str],
    split_at: int,
    n_boot: int,
    rng,
) -> Dict[str, pd.DataFrame]:
    cl = cue_long[cue_long["cue"].isin(index_cues)].copy()
    fs = first_seen(cl)
    key = pd.MultiIndex.from_frame(cl[["dkey", "cue"]])
    cl["first_seen"] = pd.to_numeric(key.map(fs.to_dict().get), errors="coerce")
    # at risk at turn t: not seen at any earlier turn; event: first seen at t
    cl["at_risk"] = cl["first_seen"].isna() | (cl["turn"] <= cl["first_seen"])
    cl["event"] = cl["turn"] == cl["first_seen"]

    # 4. first-appearance rate by turn
    risk = cl[cl["at_risk"]]
    hazard = (
        risk.groupby(["model", "cue", "turn"])
        .agg(at_risk=("dkey", "size"), new=("event", "sum"))
        .reset_index()
    )
    hazard["rate_pct"] = (hazard["new"] / hazard["at_risk"] * 100).round(2)

    # 3. early (turns 2..split_at) vs late first-appearance rate, per cue
    risk2 = risk[risk["turn"] >= 2].assign(
        period=lambda x: np.where(x["turn"] <= split_at, "early", "late")
    )
    per_d = risk2.groupby(["model", "cue", "dkey", "period"]).agg(
        at_risk=("turn", "size"), new=("event", "sum")
    )
    per_d = per_d.unstack("period", fill_value=0)
    per_d.columns = [f"{p}_{v}" for v, p in per_d.columns]
    for c in ("early_at_risk", "early_new", "late_at_risk", "late_new"):
        if c not in per_d.columns:
            per_d[c] = 0
    per_d = per_d.reset_index()
    rate_rows = []
    for (model, cue), g in per_d.groupby(["model", "cue"], sort=True):
        e, elo, ehi = boot_ratio(g["early_new"], g["early_at_risk"], n_boot, rng)
        l, llo, lhi = boot_ratio(g["late_new"], g["late_at_risk"], n_boot, rng)
        dd, dlo, dhi = boot_ratio_diff(
            g["early_new"], g["early_at_risk"], g["late_new"], g["late_at_risk"], n_boot, rng
        )
        rate_rows.append(
            dict(
                model=model, cue=cue, n_dialogues=len(g),
                early_rate_pct=_r(e * 100), early_ci95_low=_r(elo * 100), early_ci95_high=_r(ehi * 100),
                late_rate_pct=_r(l * 100), late_ci95_low=_r(llo * 100), late_ci95_high=_r(lhi * 100),
                diff_pp=_r(dd * 100), diff_ci95_low=_r(dlo * 100), diff_ci95_high=_r(dhi * 100),
            )
        )
    rates = pd.DataFrame(rate_rows)

    # 2. late onset per cue: reached turn split_at+1, cue validly rated early
    #    and late, not seen in turns 1..split_at
    per_dc = (
        cl.assign(
            _early=(cl["window"] == "early").astype(int),
            _late=(cl["window"] == "late").astype(int),
        )
        .groupby(["model", "dkey", "cue"])
        .agg(
            n_late_rated=("n_late_rated", "first"),
            early_valid=("_early", "sum"),
            late_valid=("_late", "sum"),
            first_seen=("first_seen", "first"),
        )
        .reset_index()
    )
    elig = per_dc[
        (per_dc["n_late_rated"] >= 1)
        & (per_dc["early_valid"] >= 1)
        & (per_dc["late_valid"] >= 1)
        & (per_dc["first_seen"].isna() | (per_dc["first_seen"] > split_at))
    ].copy()
    elig["late_onset"] = (elig["first_seen"] > split_at).astype(int)
    onset_rows = []
    for (model, cue), g in elig.groupby(["model", "cue"], sort=True):
        p, lo, hi = boot_ratio(g["late_onset"], np.ones(len(g)), n_boot, rng)
        onset_rows.append(
            dict(model=model, cue=cue, n_eligible=len(g), late_onset_pct=_r(p * 100),
                 ci95_low=_r(lo * 100), ci95_high=_r(hi * 100))
        )
    onset = pd.DataFrame(onset_rows)

    # 1. headline per model: any index cue first appearing after split_at
    dl = cl.groupby(["model", "dkey"]).agg(n_late_rated=("n_late_rated", "first")).reset_index()
    new_late = (
        per_dc[per_dc["first_seen"] > split_at].groupby(["model", "dkey"]).size().rename("new_types")
    )
    dl = dl.merge(new_late.reset_index(), on=["model", "dkey"], how="left")
    dl["new_types"] = dl["new_types"].fillna(0).astype(int)
    dl = dl[dl["n_late_rated"] >= 1]
    dl["any_new"] = (dl["new_types"] > 0).astype(int)
    dl["late_turns_group"] = pd.cut(
        dl["n_late_rated"], bins=[0, 2, 4, 100], labels=["1-2", "3-4", "5+"]
    ).astype(str)
    head_rows = []
    for model, g in dl.groupby("model", sort=True):
        for group_label, sub in [("all", g)] + [
            (lab, g[g["late_turns_group"] == lab]) for lab in ("1-2", "3-4", "5+")
        ]:
            if sub.empty:
                continue
            p, lo, hi = boot_ratio(sub["any_new"], np.ones(len(sub)), n_boot, rng)
            m, mlo, mhi = boot_ratio(sub["new_types"], np.ones(len(sub)), n_boot, rng)
            head_rows.append(
                dict(model=model, late_turns=group_label, n_dialogues=len(sub),
                     any_new_cue_pct=_r(p * 100), any_ci95_low=_r(lo * 100), any_ci95_high=_r(hi * 100),
                     mean_new_cue_types=_r(m, 3), mean_ci95_low=_r(mlo, 3), mean_ci95_high=_r(mhi, 3))
            )
    headline = pd.DataFrame(head_rows)

    # person-period table for the first-appearance model in R: at-risk rows
    # from turn 2 on (turn 1 is the starting prevalence, not emergence)
    risk_table = risk.loc[risk["turn"] >= 2, ["model", "dkey", "cue", "turn", "length"]].copy()
    risk_table["event"] = risk.loc[risk["turn"] >= 2, "event"].astype(int)
    return dict(headline=headline, onset=onset, rates=rates, hazard=hazard, risk_table=risk_table)


# ---------------------------------------------------------------------------
# Section 7: persistence, descriptive companion
# ---------------------------------------------------------------------------


def persistence_descriptive(cue_long: pd.DataFrame, split_at: int) -> pd.DataFrame:
    d = cue_long[cue_long["prev"].notna()].assign(
        period=lambda x: np.where(x["turn"] <= split_at, f"2-{split_at}", f"{split_at + 1}+")
    )
    out = (
        d.groupby(["model", "cue", "period", "prev"])
        .agg(n=("present", "size"), present=("present", "sum"))
        .reset_index()
    )
    out["pct_present"] = (out["present"] / out["n"] * 100).round(2)
    wide = out.pivot_table(
        index=["model", "cue", "period"], columns="prev", values=["pct_present", "n"]
    )
    wide.columns = [
        f"{'p_present' if v == 'pct_present' else 'n'}_after_{'present' if p == 1 else 'absent'}"
        for v, p in wide.columns
    ]
    return wide.reset_index()


# ---------------------------------------------------------------------------
# Section 8: pronoun use and the between-model table
# ---------------------------------------------------------------------------


def pronoun_summary(ml: pd.DataFrame, pronoun: Optional[str], split_at: int) -> pd.DataFrame:
    if not pronoun:
        return pd.DataFrame()
    p = ml[ml["measure"] == pronoun]
    rows = []
    for model, g in p.groupby("model", sort=True):
        early = g[g["window"] == "early"]
        rows.append(
            dict(model=model, n_dialogues=g["dkey"].nunique(),
                 rate_all_turns_pct=_r(g["present"].sum() / g["valid"].sum() * 100),
                 rate_turns_1_to_split_pct=_r(early["present"].sum() / early["valid"].sum() * 100))
        )
    return pd.DataFrame(rows)


def model_level_table(
    ml: pd.DataFrame,
    gap: pd.DataFrame,
    headline: pd.DataFrame,
    pronoun_tab: pd.DataFrame,
    model_info: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = ml[ml["measure"] == INDEX]
    base = idx.groupby("model").agg(
        n_dialogues=("dkey", "nunique"), present=("present", "sum"), valid=("valid", "sum")
    )
    early = idx[idx["window"] == "early"].groupby("model")[["present", "valid"]].sum()
    tab = pd.DataFrame(
        {
            "n_dialogues": base["n_dialogues"],
            "index_rate_all_turns_pct": (base["present"] / base["valid"] * 100).round(2),
            "index_rate_early_pct": (early["present"] / early["valid"] * 100).round(2),
        }
    ).reset_index()
    g = gap[gap["measure"] == INDEX][["model", "diff_pp", "ci95_low_pp", "ci95_high_pp"]]
    tab = tab.merge(g.rename(columns={"diff_pp": "index_gap_pp", "ci95_low_pp": "gap_ci95_low",
                                      "ci95_high_pp": "gap_ci95_high"}), on="model", how="left")
    h = headline[headline["late_turns"] == "all"][["model", "any_new_cue_pct"]]
    tab = tab.merge(h.rename(columns={"any_new_cue_pct": "late_emergence_pct"}), on="model", how="left")
    if not pronoun_tab.empty:
        tab = tab.merge(pronoun_tab[["model", "rate_all_turns_pct"]].rename(
            columns={"rate_all_turns_pct": "pronoun_rate_pct"}), on="model", how="left")

    corr = pd.DataFrame()
    if model_info is not None:
        mi = model_info.copy()
        mi["model"] = mi["model"].astype(str)
        tab = tab.merge(mi, on="model", how="left")
        preds = {}
        if "release_date" in tab.columns:
            dates = pd.to_datetime(tab["release_date"], errors="coerce")
            preds["release_date"] = (dates - pd.Timestamp("1970-01-01")).dt.days
        for col in ("params_active_b", "params_total_b"):
            if col in tab.columns:
                v = pd.to_numeric(tab[col], errors="coerce")
                preds[f"log10_{col}"] = np.log10(v.where(v > 0))
        outcomes = [c for c in ("index_rate_all_turns_pct", "index_rate_early_pct", "index_gap_pp",
                                "late_emergence_pct", "pronoun_rate_pct") if c in tab.columns]
        rows = []
        for pname, pv in preds.items():
            for oc in outcomes:
                ok = pv.notna() & tab[oc].notna()
                n = int(ok.sum())
                rho = pv[ok].rank().corr(tab.loc[ok, oc].rank()) if n >= 3 else np.nan
                rows.append(dict(predictor=pname, outcome=oc, n_models=n, spearman_rho=_r(rho, 3)))
        corr = pd.DataFrame(rows)
    return tab, corr


# ---------------------------------------------------------------------------
# Drop-in fix for the strict first-N filter in the analysis module
# ---------------------------------------------------------------------------


def filter_to_first_n_rated_turns(
    df: pd.DataFrame,
    n: int,
    cue_present_columns: Sequence[str],
    turn_col: str = "turn_pair_index",
    require_min_turns: bool = False,
) -> pd.DataFrame:
    """Like filter_to_first_n_turns(), but with require_min_turns=True a
    dialogue must have n RATED turns among positions 0..n-1. The original
    checks max(turn_pair_index), so 4 rated turns plus a blank closing row
    pass as 5 turns. In run_analysis(), pass cue_present_columns as the third
    argument."""
    window = df[df[turn_col] < n]
    if require_min_turns:
        rated = window[list(cue_present_columns)].isin([0, 1]).any(axis=1)
        n_rated = rated.groupby(window["dialogue_id"]).transform("sum")
        window = window[n_rated >= n]
    return window.copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    run_specs: Sequence[Tuple[str, str]],
    output_dir: str = "turn_dynamics_results",
    overrides: Optional[str] = None,
    model_info_path: Optional[str] = None,
    pronoun_cue: str = "personal pronoun use",
    split_at: int = 5,
    max_turns: int = 10,
    min_completion: Optional[float] = 0.95,
    keep_failed: bool = False,
    length_bins: Sequence[int] = (0, 5, 7, 9, 10),
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    log: List[str] = []

    raw = load_runs(run_specs)
    cue_cols = find_cue_columns(raw)
    raw, notes, applied = apply_overrides(raw, overrides)
    log += notes
    comp = completion_report(raw)
    df, comp, notes = apply_inclusion(raw, comp, min_completion, keep_failed)
    log += notes

    names = [cue_name(c) for c in cue_cols]
    if pronoun_cue in names:
        index_cues = [n for n in names if n != pronoun_cue]
    else:
        log.append(f"Pronoun cue '{pronoun_cue}' not found; the index uses all {len(names)} cues.")
        index_cues, pronoun_cue = names, None
    index_cols = [f"{n}_present" for n in index_cues]
    log.append(f"Index cues ({len(index_cues)}): {index_cues}")

    d = prepare(df, cue_cols, split_at, max_turns)
    messages = build_messages(d, index_cols)
    cue_long = build_cue_long(d, cue_cols)
    ml = build_measure_long(messages, cue_long)

    gap = five_vs_ten(ml, split_at, n_boot, rng)
    by_turn = rates_by_turn_and_length(ml, length_bins)
    em = emergence(cue_long, index_cues, split_at, n_boot, rng)
    pers = persistence_descriptive(cue_long, split_at)
    pron = pronoun_summary(ml, pronoun_cue, split_at)
    mi = read_csv_strict(model_info_path) if model_info_path else None
    if mi is not None:
        for col in ("params_total_b", "params_active_b"):
            if col in mi.columns and mi[col].astype(str).str.match(r"^-?\d+,\d+$").any():
                raise ValueError(f"{model_info_path}: decimal commas in '{col}'. "
                                 f"Run: python check_csv_format.py \"{model_info_path}\" --fix-dir normalized")
    model_tab, corr = model_level_table(ml, gap, em["headline"], pron, mi)

    outputs = {
        "completion_report.csv": comp,
        "status_overrides_applied.csv": applied,
        "messages.csv": messages,
        "cue_long.csv": cue_long,
        "five_vs_ten.csv": gap,
        "rates_by_turn_length.csv": by_turn,
        "emergence_headline.csv": em["headline"],
        "emergence_late_onset.csv": em["onset"],
        "emergence_rates.csv": em["rates"],
        "emergence_by_turn.csv": em["hazard"],
        "emergence_risk.csv": em["risk_table"],
        "persistence_descriptive.csv": pers,
        "pronoun_summary.csv": pron,
        "model_level.csv": model_tab,
        "model_level_correlations.csv": corr,
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(output_dir, name), index=False)
    with open(os.path.join(output_dir, "analysis_settings.txt"), "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [f"runs: {list(run_specs)}", f"overrides: {overrides}", f"model_info: {model_info_path}",
                 f"split_at: {split_at}", f"max_turns: {max_turns}", f"min_completion: {min_completion}",
                 f"keep_failed: {keep_failed}", f"pronoun_cue: {pronoun_cue}", f"n_boot: {n_boot}",
                 f"seed: {seed}", ""]
                + log
            )
            + "\n"
        )
    with open(os.path.join(output_dir, "pronoun_cue.txt"), "w", encoding="utf-8") as f:
        f.write((pronoun_cue or "") + "\n")
    print("\n".join(log))
    print(f"Results in {os.path.abspath(output_dir)}")
    return {k.replace(".csv", ""): v for k, v in outputs.items()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AnthroBench turn-dynamics analysis (Python part).")
    p.add_argument("--run", action="append", required=True, help="label=path/to/rated.csv (repeat)")
    p.add_argument("--output-dir", default="turn_dynamics_results")
    p.add_argument("--overrides", default=None, help="CSV: model, dialogue_id, status")
    p.add_argument("--model-info", default=None, help="CSV: model, release_date, params_total_b, params_active_b, family")
    p.add_argument("--pronoun-cue", default="personal pronoun use")
    p.add_argument("--split-at", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--min-completion", type=float, default=0.95, help="Use -1 to disable the rule.")
    p.add_argument("--keep-failed", action="store_true", help="Sensitivity run: keep failed dialogues.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run(
        parse_run_specs(a.run),
        output_dir=a.output_dir,
        overrides=a.overrides,
        model_info_path=a.model_info,
        pronoun_cue=a.pronoun_cue,
        split_at=a.split_at,
        max_turns=a.max_turns,
        min_completion=None if a.min_completion < 0 else a.min_completion,
        keep_failed=a.keep_failed,
        n_boot=a.n_boot,
        seed=a.seed,
    )
