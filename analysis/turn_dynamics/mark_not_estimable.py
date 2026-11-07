# Rare-cue fits reported as not estimable (analysis plan, Sections 5, 7, 10).
#
# Some exploratory fits flagged by BH rest on a cue that appears in at most
# MAX_PRESENT messages of that model. With so few events the logistic model
# separates and the Wald test is meaningless (odds ratios of 1e-20 to 1e12),
# often without any warning. Such flagged fits are reported as not estimable,
# as the plan asks for rare cues; the rule was set after inspection of the R
# output (Section 10). In the main run it applies to three fits. Rare-cue fits
# that are not flagged keep their estimate and note, as "How to run" Step 7 says.
#
# For each output folder this writes cue_trends_reported.csv and
# persistence_model_reported.csv next to the R files, which stay untouched:
# the estimate of each such fit is set to missing, and BH and Holm are
# recomputed within each model over the remaining index cues, exactly as the
# R script does for a fit that cannot be estimated (missing p-values are left
# out of the family). This repeats until no flagged rare-cue fit is left.
#
# Usage:
#   python mark_not_estimable.py turn_dynamics_results turn_dynamics_keep_failed turn_dynamics_keep_after_empty

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

ALPHA = 0.05
MAX_PRESENT = 3  # a flagged fit on a cue present in this many messages or fewer is not estimable
ESTIMATE_COLS = {
    "cue_trends.csv": ["or_per_turn", "ci95_low", "ci95_high", "z", "p"],
    "persistence_model.csv": ["or_carry_over", "ci95_low", "ci95_high", "z", "p"],
}


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg over the non-missing p-values, like R's p.adjust."""
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = len(q)
    if m:
        o = np.argsort(q)
        adj = np.minimum.accumulate((q[o] * m / np.arange(1, m + 1))[::-1])[::-1]
        r = np.empty(m)
        r[o] = np.minimum(adj, 1)
        out[ok] = r
    return out


def holm(p: np.ndarray) -> np.ndarray:
    """Holm over the non-missing p-values, like R's p.adjust."""
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = len(q)
    if m:
        o = np.argsort(q)
        adj = np.maximum.accumulate(q[o] * (m - np.arange(m)))
        r = np.empty(m)
        r[o] = np.minimum(adj, 1)
        out[ok] = r
    return out


def adjust_within_model(res: pd.DataFrame) -> pd.DataFrame:
    """Same families as adjust_within_model() in turn_dynamics_models.R."""
    res = res.copy()
    res["p_bh"] = np.nan
    res["p_holm"] = np.nan
    for mod in res["model"].unique():
        i = (res["model"] == mod) & (res["family"] == "index")
        p = res.loc[i, "p"].to_numpy(float)
        res.loc[i, "p_bh"] = bh(p)
        res.loc[i, "p_holm"] = holm(p)
    res["flag_bh"] = res["p_bh"].notna() & (res["p_bh"] < ALPHA)
    res["flag_holm"] = res["p_holm"].notna() & (res["p_holm"] < ALPHA)
    return res


def read(path: str) -> pd.DataFrame:
    # round-trip parsing, so values written back are exactly those R wrote
    return pd.read_csv(path, float_precision="round_trip")


def mark(folder: str) -> List[str]:
    log = []
    cl = read(os.path.join(folder, "cue_long.csv"))
    counts = cl.groupby(["model", "cue"])["present"].agg(["sum", "size"])
    for fname, cols in ESTIMATE_COLS.items():
        before = read(os.path.join(folder, fname))
        res = before.copy()
        res["estimable"] = True
        n = pd.Series([counts["sum"].get((m, c), 0) for m, c in zip(res["model"], res["cue"])], index=res.index)
        while True:
            rare = res["estimable"] & (res["family"] == "index") & res["flag_bh"] & (n <= MAX_PRESENT)
            if not rare.any():
                break
            for i in res.index[rare]:
                mod, cue = res.at[i, "model"], res.at[i, "cue"]
                n_valid = int(counts["size"].get((mod, cue), 0))
                res.loc[i, cols] = np.nan
                res.at[i, "estimable"] = False
                old = "" if pd.isna(res.at[i, "note"]) else str(res.at[i, "note"])
                new = (f"not estimable: cue present in {int(n[i])} of {n_valid} rated messages (separation); "
                       f"estimate set aside after inspection (plan, Section 10)")
                res.at[i, "note"] = f"{old}; {new}" if old else new
                log.append(f"{fname}: {mod} / {cue} marked not estimable ({int(n[i])} of {n_valid} messages).")
            res = adjust_within_model(res)
        changed = res[res["estimable"] & ((res["flag_bh"] != before["flag_bh"]) | (res["flag_holm"] != before["flag_holm"]))]
        for r in changed.itertuples():
            log.append(f"{fname}: flag change for {r.model} / {r.cue}: BH {bool(before.loc[r.Index, 'flag_bh'])} -> "
                       f"{r.flag_bh}, Holm {bool(before.loc[r.Index, 'flag_holm'])} -> {r.flag_holm}")
        if changed.empty:
            log.append(f"{fname}: no other flag changes.")
        res.to_csv(os.path.join(folder, fname.replace(".csv", "_reported.csv")), index=False)
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Report rare-cue fits as not estimable and recompute BH/Holm.")
    p.add_argument("folders", nargs="+", help="Output folders of turn_dynamics.py + turn_dynamics_models.R")
    for folder in p.parse_args().folders:
        print(f"== {folder}")
        print("\n".join("  " + line for line in mark(folder)))
