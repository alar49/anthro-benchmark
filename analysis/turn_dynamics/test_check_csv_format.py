"""Synthetic check of check_csv_format.py and of the clear-error guard in turn_dynamics.py.

Run: python test_check_csv_format.py
"""
import contextlib
import io
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_csv_format as cf  # noqa: E402
import turn_dynamics as td  # noqa: E402

OUT = os.path.join(HERE, "synthetic_format")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "model_a"))
os.makedirs(os.path.join(OUT, "model_b"))

CUES = ["empathy", "validation", "personal pronoun use"]
rows = []
for d in range(4):
    for t in range(6):
        r = {"dialogue_id": d, "turn_pair_index": t, "dialogue_status": "completed_early_natural_end",
             "user_message": "Ciao, perché?", "assistant_message": 'Sure; "quoted", with commas, and\na newline'}
        for c in CUES:
            r[f"{c}_present"] = (d + t) % 2
        rows.append(r)
good = pd.DataFrame(rows)
p_good = os.path.join(OUT, "model_a", "rated.csv")
good.to_csv(p_good, index=False)

# Excel-style: ';' separator, Windows encoding
p_semi = os.path.join(OUT, "model_b", "rated.csv")
good.to_csv(p_semi, index=False, sep=";", encoding="cp1252")

# content errors: a duplicated turn and a value of 2
bad = pd.concat([good, good.iloc[[0]]], ignore_index=True)
bad.loc[3, "empathy_present"] = 2
p_bad = os.path.join(OUT, "rated_bad.csv")
bad.to_csv(p_bad, index=False)

p_info = os.path.join(OUT, "model_info.csv")
with open(p_info, "w", encoding="utf-8") as f:
    f.write("model;release_date;params_total_b;params_active_b;family\n"
            "model_a;03/04/2026;120;5,1;x\nmodel_b;2025-06-01;24;24;y\n")
p_ov = os.path.join(OUT, "status_overrides.csv")
pd.DataFrame({"model": ["model_a", "model_a"], "dialogue_id": [1, 1],
              "status": ["completed*", "completed*"]}).to_csv(p_ov, index=False)


def run_checker(files, fix=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cf.main(files + (["--fix-dir", fix] if fix else []))
    return code, buf.getvalue()


checks = []
code, out = run_checker([p_good])
checks.append(("good file passes", code == 0 and "[OK]" in out))

code, out = run_checker([p_semi])
checks.append(("';' + cp1252 file fails on separator and encoding",
               code == 1 and "separator is ';'" in out and "not UTF-8" in out))

try:
    with contextlib.redirect_stdout(io.StringIO()):
        td.run([("model_b", p_semi)], output_dir=os.path.join(OUT, "res_semi"), n_boot=50)
    checks.append(("turn_dynamics stops with a clear message on the ';' file", False))
except ValueError as e:
    checks.append(("turn_dynamics stops with a clear message on the ';' file", "check_csv_format.py" in str(e)))

fix_dir = os.path.join(OUT, "normalized")
code, out = run_checker([p_good, p_semi, p_info, p_ov, p_bad], fix=fix_dir)
print(out)
checks.append(("bad content reported: duplicate and value 2",
               "duplicated (dialogue_id, turn_pair_index)" in out and "values not in" in out))
checks.append(("model_info: decimal comma and date reported",
               "decimal commas in" in out and "release_date not YYYY-MM-DD" in out))
checks.append(("overrides: duplicate reported", "appears more than once" in out))
checks.append(("same-name files kept apart in the fix folder",
               os.path.exists(os.path.join(fix_dir, "rated.csv"))
               and os.path.exists(os.path.join(fix_dir, "model_b__rated.csv"))))

code, out = run_checker([os.path.join(fix_dir, "model_b__rated.csv")])
checks.append(("normalized ';' file passes", code == 0))
norm = pd.read_csv(os.path.join(fix_dir, "model_b__rated.csv"), dtype=str, keep_default_na=False)
orig = good.astype(str)
checks.append(("normalized file keeps every value (text, accents, newlines)",
               norm[orig.columns].equals(orig)))
code, out = run_checker([os.path.join(fix_dir, "model_info.csv")])
checks.append(("normalized model_info: decimal fixed, date still flagged",
               "decimal commas in" not in out and "release_date not YYYY-MM-DD" in out))

with contextlib.redirect_stdout(io.StringIO()):
    res = td.run([("model_b", os.path.join(fix_dir, "model_b__rated.csv"))],
                 output_dir=os.path.join(OUT, "res_norm"), n_boot=50)
checks.append(("turn_dynamics runs on the normalized file", len(res["messages"]) == 24))

for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
