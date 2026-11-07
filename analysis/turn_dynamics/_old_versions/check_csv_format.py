# Format checker for the CSVs used by the turn analysis.
#
# The standard every file must follow:
#   - comma-separated
#   - UTF-8 text (a byte-order mark is tolerated)
#   - decimal point, not decimal comma (0.5, not 0,5)
#   - dates as YYYY-MM-DD
#
# Files saved from Excel with Italian or other European settings often use
# ';' as separator, ',' as decimal mark, a Windows encoding and dd/mm/yyyy
# dates. pandas then reads the whole row as one column and the analysis stops.
#
# The checker recognises three kinds of file by their columns and runs the
# matching content checks:
#   rated       dialogue_id, turn_pair_index, <cue>_present ...
#   overrides   model, dialogue_id, status
#   model_info  model, release_date, params_total_b, params_active_b, family
#
# Usage:
#   python check_csv_format.py results/*/rated.csv status_overrides.csv model_info.csv
#   python check_csv_format.py <files> --fix-dir normalized
#
# --fix-dir writes a standard copy of every readable file into that folder
# (comma-separated, UTF-8, decimal points in numeric columns); the originals
# are never changed. Dates are reported, never rewritten, because 03/04/2026
# could be either day/month or month/day.
#
# Exit code 0 = no errors (warnings allowed), 1 = at least one error, so the
# checker can gate a run script.

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

CANDIDATE_SEPS = [",", ";", "\t"]
DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OVERRIDE_COLS = {"model", "dialogue_id", "status"}
MODEL_INFO_COLS = {"model", "release_date", "params_total_b", "params_active_b", "family"}


def detect_encoding(path: str) -> Tuple[str, bool]:
    with open(path, "rb") as f:
        raw = f.read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        raw.decode("utf-8")
        return "utf-8-sig", bom
    except UnicodeDecodeError:
        return "cp1252", bom


def cue_columns(cols: List[str]) -> List[str]:
    """Same selection rule as the analysis module."""
    return [c for c in cols if c.endswith("_present") and "_s" not in c and "_final_present" not in c]


def detect_separator(path: str, encoding: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """Tries each candidate separator on the whole file and keeps the one that
    gives known columns, or the most columns."""
    best: Tuple[int, Optional[str], Optional[pd.DataFrame]] = (-1, None, None)
    for sep in CANDIDATE_SEPS:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding=encoding)
        except Exception:
            continue
        cols = set(df.columns)
        known = int({"dialogue_id", "turn_pair_index"} <= cols) * 100 + len(cols & (OVERRIDE_COLS | MODEL_INFO_COLS)) * 10
        score = known + len(df.columns)
        if len(df.columns) > 1 and score > best[0]:
            best = (score, sep, df)
    return best[1], best[2]


def file_kind(cols: List[str]) -> str:
    s = set(cols)
    if {"dialogue_id", "turn_pair_index"} <= s:
        return "rated"
    if OVERRIDE_COLS <= s:
        return "overrides"
    if {"model", "release_date"} <= s:
        return "model_info"
    return "unknown"


def _as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.str.strip().str.replace(",", ".", regex=False), errors="coerce")


def check_rated(df: pd.DataFrame, errors: List[str], warnings: List[str]) -> List[str]:
    numeric = ["turn_pair_index"]
    cues = cue_columns(list(df.columns))
    if not cues:
        errors.append("no '<cue>_present' columns")
    elif len(cues) != 14:
        warnings.append(f"{len(cues)} cue columns (expected 14)")
    numeric += cues
    t = _as_number(df["turn_pair_index"])
    if t.isna().any() or (t < 0).any() or (t % 1 != 0).any():
        errors.append("turn_pair_index has empty, negative or non-integer values")
    for c in cues:
        v = _as_number(df[c])
        bad = v.isna() | ~v.isin([-1, 0, 1])
        if bad.any():
            errors.append(f"'{c}': {int(bad.sum())} values not in {{-1, 0, 1}}")
    if df["dialogue_id"].str.strip().eq("").any():
        errors.append("empty dialogue_id values")
    dup = df.duplicated(["dialogue_id", "turn_pair_index"]).sum()
    if dup:
        errors.append(f"{int(dup)} duplicated (dialogue_id, turn_pair_index) rows, e.g. a re-run appended without removing the failed attempt")
    if "dialogue_status" in df.columns:
        n_status = df.groupby("dialogue_id")["dialogue_status"].nunique()
        if (n_status > 1).any():
            warnings.append(f"{int((n_status > 1).sum())} dialogues with more than one dialogue_status")
    else:
        warnings.append("no dialogue_status column: the 95% rule and the failed-dialogue exclusion cannot be applied")
    if "target_llm" in df.columns and df["target_llm"].nunique() > 1:
        warnings.append(f"{df['target_llm'].nunique()} different target_llm values: one model per file expected")
    if df["dialogue_id"].str.fullmatch(r"\d+\.0").any():
        warnings.append("dialogue_id values like '12.0': the file may have been re-saved by a spreadsheet; ids must match the original")
    return numeric


def check_overrides(df: pd.DataFrame, errors: List[str], warnings: List[str]) -> List[str]:
    if df.duplicated(["model", "dialogue_id"]).any():
        errors.append("the same (model, dialogue_id) appears more than once")
    st = df["status"].str.strip()
    odd = ~(st.str.startswith("completed") | st.str.startswith("failed_at_turn_") | st.str.startswith("stopped_budget_exceeded_"))
    if odd.any():
        warnings.append(f"{int(odd.sum())} statuses not starting with completed / failed_at_turn_ / stopped_budget_exceeded_")
    return []


def check_model_info(df: pd.DataFrame, errors: List[str], warnings: List[str]) -> List[str]:
    if df["model"].duplicated().any():
        errors.append("the same model appears more than once")
    dates = df["release_date"].str.strip()
    bad = (dates != "") & ~dates.str.match(ISO_DATE)
    if bad.any():
        errors.append(f"release_date not YYYY-MM-DD in {int(bad.sum())} rows, e.g. '{dates[bad].iloc[0]}' (not rewritten: day/month order is ambiguous)")
    numeric = [c for c in ("params_total_b", "params_active_b") if c in df.columns]
    for c in numeric:
        v = df[c].str.strip()
        bad = (v != "") & _as_number(df[c]).isna()
        if bad.any():
            errors.append(f"'{c}' has non-numeric values, e.g. '{v[bad].iloc[0]}'")
    return numeric


def check_file(path: str, fix_dir: Optional[str]) -> Dict:
    errors: List[str] = []
    warnings: List[str] = []
    encoding, bom = detect_encoding(path)
    if encoding != "utf-8-sig":
        errors.append("not UTF-8 (looks like a Windows/Excel encoding)")
    sep, df = detect_separator(path, encoding)
    if df is None:
        errors.append("could not be read with ',', ';' or tab as separator")
        return dict(file=path, kind="?", separator="?", rows=0, errors=errors, warnings=warnings, fixed=None)
    if sep != ",":
        errors.append(f"separator is '{'TAB' if sep == chr(9) else sep}', not ','")
    df.columns = [c.strip() for c in df.columns]
    kind = file_kind(list(df.columns))
    if kind == "rated":
        numeric = check_rated(df, errors, warnings)
    elif kind == "overrides":
        numeric = check_overrides(df, errors, warnings)
    elif kind == "model_info":
        numeric = check_model_info(df, errors, warnings)
    else:
        numeric = []
        warnings.append("unknown file type: only the format was checked")
    comma_cols = [c for c in numeric if df[c].str.strip().str.match(DECIMAL_COMMA).any()]
    if comma_cols:
        errors.append(f"decimal commas in {comma_cols}")

    fixed = None
    if fix_dir:
        os.makedirs(fix_dir, exist_ok=True)
        out = df.copy()
        for c in numeric:
            mask = out[c].str.strip().str.match(DECIMAL_COMMA)
            out.loc[mask, c] = out.loc[mask, c].str.strip().str.replace(",", ".", regex=False)
        fixed = os.path.join(fix_dir, os.path.basename(path))
        if os.path.abspath(fixed) == os.path.abspath(path):
            raise ValueError("--fix-dir must be a different folder from the input files")
        if os.path.exists(fixed):  # several files with the same name, e.g. results/<model>/rated.csv
            parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
            fixed = os.path.join(fix_dir, f"{parent}__{os.path.basename(path)}")
        out.to_csv(fixed, index=False, encoding="utf-8")
    return dict(file=path, kind=kind, separator="TAB" if sep == "\t" else sep, rows=len(df),
                errors=errors, warnings=warnings, fixed=fixed)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check (and optionally normalize) CSV format for the turn analysis.")
    p.add_argument("files", nargs="+", help="CSV files or glob patterns")
    p.add_argument("--fix-dir", default=None, help="Write standard copies here; originals are untouched")
    a = p.parse_args(argv)
    paths: List[str] = []
    for pattern in a.files:
        matches = sorted(glob.glob(pattern))
        paths += matches if matches else [pattern]
    n_err = 0
    for path in paths:
        if not os.path.isfile(path):
            print(f"\n{path}\n  ERROR  file not found")
            n_err += 1
            continue
        r = check_file(path, a.fix_dir)
        status = "OK" if not r["errors"] else "FAIL"
        print(f"\n[{status}] {path}  (type: {r['kind']}, separator: {r['separator']}, rows: {r['rows']})")
        for e in r["errors"]:
            print(f"  ERROR    {e}")
        for w in r["warnings"]:
            print(f"  WARNING  {w}")
        if r["fixed"]:
            print(f"  standard copy written: {r['fixed']}")
        n_err += bool(r["errors"])
    print(f"\n{len(paths)} files checked, {n_err} with errors.")
    if n_err and not a.fix_dir:
        print("Format errors (separator, encoding, decimal commas) can be fixed with --fix-dir <folder>;"
              " content errors (duplicates, wrong values, dates) need a manual fix.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
