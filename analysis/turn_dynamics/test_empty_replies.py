"""Synthetic check of the empty-reply handling in turn_dynamics.py.

Five dialogues of 8 turns. The target returns a blank answer (one token, a
space) at:
  dialogue 1, turn 1          -> no usable turn left
  dialogue 2, turn 4          -> 3 usable turns, 4 later answered turns dropped
  dialogue 3, turns 3 and 4   -> 2 usable turns, 4 later answered turns dropped
Dialogue 4 ends naturally after 6 turns with a blank closing row where the
target was NOT called (no tokens): not an empty reply.

Run: python test_empty_replies.py
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import turn_dynamics as td  # noqa: E402

OUT = os.path.join(HERE, "synthetic_empty")
os.makedirs(OUT, exist_ok=True)
CUES = ["empathy", "validation", "personal pronoun use"]
empty_at = {(1, 0), (2, 3), (3, 2), (3, 3)}
rows = []
for d in range(5):
    n = 6 if d == 4 else 8
    status = "completed_early_natural_end" if d == 4 else "completed"
    for t in range(n):
        r = {"dialogue_id": d, "turn_pair_index": t, "dialogue_status": status, "user_message": "u",
             "assistant_message": "a", "target_output_tokens": 200}
        for c in CUES:
            r[f"{c}_present"] = int((d + t) % 3 == 0)
        if (d, t) in empty_at:
            r["assistant_message"] = " "
            r["target_output_tokens"] = 1
            for c in CUES:
                r[f"{c}_present"] = -1
        rows.append(r)
    if d == 4:  # blank closing row, target not called
        r = {"dialogue_id": d, "turn_pair_index": n, "dialogue_status": status, "user_message": "bye",
             "assistant_message": np.nan, "target_output_tokens": np.nan}
        for c in CUES:
            r[f"{c}_present"] = -1
        rows.append(r)
path = os.path.join(OUT, "rated.csv")
pd.DataFrame(rows).to_csv(path, index=False)

with contextlib.redirect_stdout(io.StringIO()):
    cut = td.run([("m", path)], output_dir=os.path.join(OUT, "cut"), n_boot=50, min_completion=None)
    keep = td.run([("m", path)], output_dir=os.path.join(OUT, "keep"), n_boot=50, min_completion=None,
                  keep_after_empty_reply=True)

e = cut["empty_replies"].set_index("dialogue_id")
comp = cut["completion_report"].iloc[0]
msgs = cut["messages"]
checks = [
    ("three dialogues with empty replies", len(e) == 3),
    ("closing row without a target call is not an empty reply", 4 not in e.index),
    ("first empty turns 1, 4, 3", e.loc[[1, 2, 3], "first_empty_turn"].tolist() == [1, 4, 3]),
    ("answered turns after: 7, 4, 4", e.loc[[1, 2, 3], "rated_turns_after"].tolist() == [7, 4, 4]),
    ("completion report: 1 dialogue without usable turns", comp["dialogues_without_usable_turns"] == 1),
    ("usable rate 4/5", abs(comp["usable_rate"] - 0.8) < 1e-9),
    ("cut: dialogue 1 gone, dialogue 2 keeps turns 1-3",
     "m::1" not in set(msgs["dkey"]) and sorted(msgs.loc[msgs["dkey"] == "m::2", "turn"]) == [1, 2, 3]),
    ("keep: later turns stay (dialogue 2 has 7 rated turns)",
     (keep["messages"]["dkey"] == "m::2").sum() == 7),
]
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
