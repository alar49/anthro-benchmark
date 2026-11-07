"""Synthetic check of turn_dynamics.py.

Builds rated CSVs with the same layout as AnthroBench (one per model), with
known effects, and checks that the Python outputs recover them:

  null_sel  no change inside any dialogue, but warmer dialogues run longer
            (the selection effect): the old pooled 5-vs-10 comparison shows
            an increase, the same-dialogue comparison should not
  up        a real within-dialogue increase
  carry     no trend, but cues persist from turn t-1 to t
  lowcomp   8% failed dialogues: must be excluded by the 95% rule

Also checks the status override (a failed dialogue relabeled as finished),
blank closing rows, per-sample '_s1' columns, and missing (-1) ratings.

Run: python test_turn_dynamics.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import turn_dynamics as td  # noqa: E402

rng = np.random.default_rng(3)
OUT = os.path.join(HERE, "synthetic")
os.makedirs(OUT, exist_ok=True)

INDEX_CUES = ["desires", "emotions", "agency", "personal history", "personal relationships",
              "sentience", "physical embodiment", "movement and interactions", "sensory input",
              "empathy", "validation", "relatability", "explicit relationship status"]
PRONOUN = "personal pronoun use"
BASE = dict(zip(INDEX_CUES, np.linspace(-3.0, -0.5, len(INDEX_CUES))))


def simulate(model, n_dial, beta, gamma, n_failed=0, start_id=0):
    rows = []
    for i in range(n_dial):
        did = start_id + i
        u = rng.normal(0, 1)
        length = int(np.clip(round(7.5 + 2 * u), 5, 10))  # warmer dialogues run longer
        status = "completed" if length == 10 else "completed_early_natural_end"
        if i < n_failed:
            length = int(rng.integers(3, 8))
            status = f"failed_at_turn_{length + 1}"
        prev = {c: 0 for c in INDEX_CUES}
        for t in range(length):
            r = {"dialogue_id": did, "turn_pair_index": t, "dialogue_status": status,
                 "user_message": "u", "assistant_message": "a", "target_llm": model}
            for c in INDEX_CUES:
                eta = BASE[c] + u + beta * (t + 1 - 5.5) + gamma * prev[c]
                v = int(rng.random() < 1 / (1 + np.exp(-eta)))
                prev[c] = v
                r[f"{c}_present"] = -1 if rng.random() < 0.02 else v
                r[f"{c}_s1_present"] = v  # per-sample column, must be ignored
            r[f"{PRONOUN}_present"] = int(rng.random() < 0.95)
            rows.append(r)
        if status == "completed_early_natural_end" and rng.random() < 0.5:  # blank closing row
            r = {"dialogue_id": did, "turn_pair_index": length, "dialogue_status": status,
                 "user_message": "", "assistant_message": "", "target_llm": model}
            for c in INDEX_CUES + [PRONOUN]:
                r[f"{c}_present"] = -1
                r[f"{c}_s1_present"] = -1
            rows.append(r)
    return pd.DataFrame(rows)


specs = []
for model, beta, gamma, n_failed in [("null_sel", 0.0, 0.0, 2), ("up", 0.15, 0.0, 3),
                                     ("carry", 0.0, 1.2, 0), ("lowcomp", 0.0, 0.0, 8)]:
    df = simulate(model, 100, beta, gamma, n_failed)
    path = os.path.join(OUT, f"rated_{model}.csv")
    df.to_csv(path, index=False)
    specs.append((model, path))

# one failed dialogue in "up" was "verified as finished"
pd.DataFrame({"model": ["up"], "dialogue_id": [0], "status": ["completed_early_natural_end*"]}).to_csv(
    os.path.join(OUT, "status_overrides.csv"), index=False)
pd.DataFrame({"model": ["null_sel", "up", "carry", "lowcomp"],
              "release_date": ["2024-01-15", "2025-03-01", "2025-09-10", "2024-06-01"],
              "params_total_b": [8, 70, 120, 4], "params_active_b": [8, 70, 5.1, 4],
              "family": ["a", "b", "c", "d"]}).to_csv(os.path.join(OUT, "model_info.csv"), index=False)

# --- Old pooled comparison (first 5 vs all turns), 13-cue index ---
print("Old pooled comparison, 13-cue index (first 5 turns vs all turns):")
for model, path in specs[:3]:
    raw = pd.read_csv(path)
    cols = [f"{c}_present" for c in INDEX_CUES]
    def rate(x):
        v = x[cols].to_numpy().ravel()
        v = v[(v == 0) | (v == 1)]
        return v.mean() * 100
    print(f"  {model:9s} first5 {rate(raw[raw.turn_pair_index < 5]):5.1f}%   all {rate(raw):5.1f}%")

res = td.run(specs, output_dir=os.path.join(OUT, "results"),
             overrides=os.path.join(OUT, "status_overrides.csv"),
             model_info_path=os.path.join(OUT, "model_info.csv"), n_boot=500)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
print("\nCompletion report:\n", res["completion_report"].to_string(index=False))

gap = res["five_vs_ten"]
print("\nSame-dialogue 5-vs-10 gap, 13-cue index:")
print(gap[gap.measure == td.INDEX].to_string(index=False))
print("\nEmergence headline (all late-turn groups):")
h = res["emergence_headline"]
print(h[h.late_turns == "all"].to_string(index=False))
print("\nPersistence (descriptive), 'empathy':")
p = res["persistence_descriptive"]
print(p[p.cue == "empathy"].to_string(index=False))
print("\nPronoun summary:\n", res["pronoun_summary"].to_string(index=False))
print("\nModel-level table:\n", res["model_level"].to_string(index=False))

# --- checks ---
checks = []
models_in = set(res["messages"]["model"])
checks.append(("lowcomp excluded by the 95% rule", "lowcomp" not in models_in))
checks.append(("override kept dialogue 0 of 'up'", "up::0" in set(res["messages"]["dkey"])))
checks.append(("blank closing rows are not turns", res["messages"]["n"].min() > 0))
checks.append(("index has 13 cues", res["messages"]["n"].max() == 13))
g = gap[gap.measure == td.INDEX].set_index("model")
checks.append(("null_sel: same-dialogue gap CI includes 0",
               g.loc["null_sel", "ci95_low_pp"] <= 0 <= g.loc["null_sel", "ci95_high_pp"]))
checks.append(("up: same-dialogue gap CI above 0", g.loc["up", "ci95_low_pp"] > 0))
pe = p[(p.cue == "empathy")].groupby("model")[["p_present_after_present", "p_present_after_absent"]].mean()
checks.append(("carry: P(present | prev present) > P(present | prev absent)",
               pe.loc["carry", "p_present_after_present"] > pe.loc["carry", "p_present_after_absent"]))
cl = res["cue_long"]
ok_prev = cl.dropna(subset=["prev"])
checks.append(("prev only defined from turn 2 on", ok_prev["turn"].min() >= 2))
print()
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
