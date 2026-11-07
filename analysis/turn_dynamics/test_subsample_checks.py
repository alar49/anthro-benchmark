"""Synthetic check of subsample_checks.py.

all960: 960 scenarios, one dialogue each, scenario-level differences.
seed42: 96 of them, the same rows (an a-posteriori extraction).
test:   a second run on the same 96 scenarios, different dialogue ids.
biased: 96 of the warmest scenarios, to confirm a real difference is caught.

Run: python test_subsample_checks.py
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import subsample_checks as sc  # noqa: E402

rng = np.random.default_rng(11)
OUT = os.path.join(HERE, "synthetic_checks")
os.makedirs(OUT, exist_ok=True)
CUES = ["desires", "emotions", "agency", "empathy", "validation", "relatability"]
PRONOUN = "personal pronoun use"
scen_effect = rng.normal(0, 1, 960)


def gen(scenarios, id_offset, label):
    rows = []
    for j, s in enumerate(scenarios):
        u = scen_effect[s] + rng.normal(0, 0.5)  # scenario effect + run-specific noise
        length = int(np.clip(round(7.5 + 2 * u), 5, 10))
        status = "completed" if length == 10 else "completed_early_natural_end"
        for t in range(length):
            r = {"dialogue_id": id_offset + j, "scenario_id": f"s{s}", "turn_pair_index": t,
                 "dialogue_status": status, "target_llm": label}
            for k, c in enumerate(CUES):
                p = 1 / (1 + np.exp(-(-2 + 0.3 * k + u + 0.1 * (t + 1 - 5.5))))
                r[f"{c}_present"] = int(rng.random() < p)
            r[f"{PRONOUN}_present"] = int(rng.random() < 0.95)
            rows.append(r)
    return pd.DataFrame(rows)


full = gen(range(960), 0, "all960")
seed_scen = rng.choice(960, 96, replace=False)
seed_ids = set(seed_scen)                       # dialogue_id == scenario index in the full run
seed42 = full[full["dialogue_id"].isin(seed_ids)]
test = gen(seed_scen, 100000, "test")
warm = np.argsort(scen_effect)[-96:]
biased = full[full["dialogue_id"].isin(set(warm))]

paths = {}
for name, frame in [("full", full), ("seed42", seed42), ("test", test), ("biased", biased)]:
    paths[name] = os.path.join(OUT, f"{name}.csv")
    frame.to_csv(paths[name], index=False)

with contextlib.redirect_stdout(io.StringIO()):
    res = sc.run(("ms4_all960", paths["full"]), ("ms4_seed42", paths["seed42"]), ("ms4_test", paths["test"]),
                 output_dir=os.path.join(OUT, "results"), scenario_col="scenario_id", n_boot=500)
    res_b = sc.run(("ms4_all960", paths["full"]), ("ms4_biased", paths["biased"]), None,
                   output_dir=os.path.join(OUT, "results_biased"), n_boot=500)

pd.set_option("display.width", 220)
c1 = res["subsample_vs_rest"]
c2 = res["test_vs_seed42_paired"]
cb = res_b["subsample_vs_rest"]
print("Check 1 (random subsample), index:")
print(c1[c1.measure.isin(["index_13", "dialogue"])].to_string(index=False))
print("\nCheck 2 (paired), index:")
print(c2[c2.measure.isin(["index_13", "dialogue"])].to_string(index=False))
print("\nCheck 1 (biased subsample), index:")
print(cb[cb.measure.isin(["index_13", "dialogue"])].to_string(index=False))
print(open(os.path.join(OUT, "results", "checks_log.txt")).read())

i1 = c1[(c1.measure == "index_13") & (c1.statistic == "rate_all_pct")].iloc[0]
i2 = c2[(c2.measure == "index_13") & (c2.statistic == "rate_all_pct")].iloc[0]
ib = cb[(cb.measure == "index_13") & (cb.statistic == "rate_all_pct")].iloc[0]
checks = [
    ("check 1 split: 864 dialogues in the rest", i1.n_rest == 864),
    ("check 1, random subsample: 95% CI includes 0", i1.ci95_low <= 0 <= i1.ci95_high),
    ("check 1, biased subsample: 95% CI above 0", ib.ci95_low > 0),
    ("check 1, biased subsample: not equivalent", not ib.equivalent_within_margin),
    ("check 2: 96 scenarios paired", i2.n_scenarios == 96),
    ("check 2: paired 95% CI includes 0", i2.ci95_low <= 0 <= i2.ci95_high),
    ("check 2: positive correlation across scenarios", i2.spearman_across_scenarios > 0.3),
]
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
