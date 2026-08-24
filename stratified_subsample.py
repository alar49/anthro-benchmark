"""
Stratified (balanced) subsampling for AnthroBench's first_turns.csv
(Ibrahim et al., 2025, "Multi-turn Evaluation of Anthropomorphic Behaviours
in Large Language Models", arXiv:2502.07077).

DATASET STRUCTURE (verified against the actual file):
  960 rows = 120 hand-crafted base prompts x 4 use_domains x 2 use_scenarios/domain.
  - empathy, professionalism are fully DETERMINED by use_domain (not free variables).
  - use_scenario is nested in use_domain (2 per domain, no overlap).
  - behavior_category is fully DETERMINED by cue (12 cues, 3 per category).
  - original_prompt is nested in cue (10 base prompts per cue) and CROSSED with
    use_domain x use_scenario (each of the 120 prompts appears in all 8
    domain-scenario cells exactly once -> a complete, unreplicated factorial).

Because of this structure there are only two independent stratifying axes:
  (1) cue  (which fixes behavior_category)
  (2) use_domain  (which fixes empathy/professionalism), crossed with use_scenario

IMPLICATION FOR SAMPLING:
  original_prompt only has 1 row per (domain, scenario) cell, so it cannot be
  proportionally "sampled" within a cell -- it can only be included or excluded.
  Naive row-level stratified sampling on the (domain, scenario, cue) grid will
  therefore fragment base prompts (keep a prompt's "friendship" version but
  drop its "career development" version). To avoid that, treat original_prompt
  itself as the sampling unit, stratified by cue, and keep ALL context
  replicates of every prompt you select. This is `block_stratified_sample`
  below, and is the recommended default.

  `row_stratified_sample` is provided as a simpler alternative for cases where
  you explicitly don't mind fragmenting prompts (e.g. you only care about
  domain/scenario/cue proportions, not about within-prompt cross-domain
  comparisons), and/or you need an arbitrary n not divisible by the design.

PATCH NOTE (this version): `row_stratified_sample` previously reused the same
literal `random_state` for every stratum's `.sample()` call, which collapses
same-sized strata onto near-identical row selections (see that function's
docstring for the verified numbers). Fixed by deriving a per-stratum seed,
the same pattern `build_balanced_sample.py` already uses. See
docs/sampling_strategies_guide.md for the full write-up, including a
head-to-head comparison against `build_balanced_sample.py`.
"""

import argparse
import os

import pandas as pd
import numpy as np


def validate_structure(df: pd.DataFrame) -> None:
    """Fail loudly if the assumed nesting/crossing structure doesn't hold
    (e.g. if you point this at a different export of the benchmark)."""
    assert df.groupby('cue')['behavior_category'].nunique().max() == 1, \
        "cue is not nested in behavior_category"
    assert df.groupby('use_domain')[['empathy', 'professionalism']].nunique().max().max() == 1, \
        "empathy/professionalism are not fully determined by use_domain"
    assert df.groupby('use_domain')['use_scenario'].nunique().eq(2).all(), \
        "use_scenario is not exactly 2-per-domain"
    assert df.groupby('cue')['original_prompt'].nunique().eq(10).all(), \
        "cue does not have exactly 10 base prompts"
    assert df.groupby('original_prompt')['use_domain'].nunique().eq(4).all(), \
        "original_prompt is not crossed with all 4 use_domains"


def block_stratified_sample(df, k_per_cue=None, frac=None, cue_col='cue',
                             prompt_col='original_prompt', random_state=42):
    """RECOMMENDED. Sampling unit = original_prompt, stratified by cue.
    Keeps every selected prompt's full set of context rows (all domain x
    scenario replicates), so use_domain, use_scenario, empathy,
    professionalism, cue and behavior_category proportions are preserved
    EXACTLY, and no prompt ends up only partially represented.

    Pass exactly one of k_per_cue (int, prompts to keep per cue, max 10) or
    frac (float in (0,1], fraction of each cue's 10 prompts to keep).
    Valid resulting sample sizes are multiples of 96 (12 cues x 8 context cells).
    """
    assert (k_per_cue is None) != (frac is None), "pass exactly one of k_per_cue or frac"
    rng = np.random.default_rng(random_state)
    keep_prompts = []
    for cue, sub in df.groupby(cue_col):
        prompts = sub[prompt_col].unique()
        k = k_per_cue if k_per_cue is not None else round(frac * len(prompts))
        k = max(0, min(k, len(prompts)))
        keep_prompts.extend(rng.choice(prompts, size=k, replace=False))
    return df[df[prompt_col].isin(keep_prompts)].copy()


def row_stratified_sample(df, strata_cols=('use_domain', 'use_scenario', 'cue'),
                           n=None, frac=None, random_state=42):
    """Alternative. Simple row-level proportional stratified sample on the
    given strata columns, hitting an arbitrary n via largest-remainder
    apportionment. Does NOT guarantee intact original_prompt coverage --
    a given base prompt may end up represented in some domains/scenarios
    but not others.

    PATCHED: each stratum now draws with its own derived seed
    (random_state + its position in the stable, sorted stratum order)
    instead of the same literal random_state reused as-is for every
    stratum. Reusing one random_state across same-sized strata makes
    pandas' .sample() pick the same *relative* row positions in every one
    of them -- verified on the real first_turns.csv: with the old code,
    n=300 allocates k=3 to 84 of the 96 (domain, scenario, cue) strata,
    and all 84 collapsed to just 12 distinct prompt-selection patterns
    (one per cue) instead of up to 84 independent draws. This is the same
    failure mode already fixed in build_balanced_sample.py via
    group_seed = seed + group_position; row_stratified_sample just hadn't
    received the equivalent fix until now. block_stratified_sample was
    never affected -- it draws from a single numpy Generator whose state
    advances across cues, rather than reseeding pandas per call.
    """
    assert (n is None) != (frac is None), "pass exactly one of n or frac"
    strata_cols = list(strata_cols)
    sizes = df.groupby(strata_cols).size()
    if frac is not None:
        alloc = (sizes * frac).round().astype(int).clip(upper=sizes)
    else:
        raw = sizes / sizes.sum() * n
        alloc = np.floor(raw).astype(int)
        remainder = int(n - alloc.sum())
        order = (raw - alloc).sort_values(ascending=False).index
        for key in order[:remainder]:
            alloc[key] += 1
    parts = []
    for position, (key, k) in enumerate(alloc.items()):
        if k <= 0:
            continue
        mask = np.ones(len(df), dtype=bool)
        keys = key if isinstance(key, tuple) else (key,)
        for col, val in zip(strata_cols, keys):
            mask &= (df[col] == val)
        stratum_seed = random_state + position
        parts.append(df[mask].sample(n=int(k), random_state=stratum_seed))
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def check_balance(df_full, df_sub, cols=('use_domain', 'use_scenario', 'empathy',
                                          'professionalism', 'cue', 'behavior_category')):
    """Print how closely the subset's proportions match the full dataset's,
    per column, plus original_prompt coverage."""
    print(f"n = {len(df_sub)} ({len(df_sub) / len(df_full):.1%} of full data), "
          f"unique original_prompt = {df_sub['original_prompt'].nunique()} / "
          f"{df_full['original_prompt'].nunique()}")
    for col in cols:
        p_full = df_full[col].value_counts(normalize=True).sort_index()
        p_sub = df_sub[col].value_counts(normalize=True).reindex(p_full.index, fill_value=0)
        print(f"  {col:20s} max |proportion diff| = {(p_full - p_sub).abs().max():.4f}")


def _default_input_path() -> str:
    """Resolve first_turns.csv without assuming any particular environment.

    Priority:
    1. './first_turns.csv' in the current working directory -- covers
       Colab/Kaggle, where you've typically just uploaded or !wget'ed it
       there, and any local run from a directory you've already put it in.
    2. The copy bundled inside an installed anthro_benchmark package (only
       resolves if you've run `pip install -e .` from the repo root) --
       the same file DialogueGenerator._load_prompts() reads via
       importlib.resources when no --custom-prompt-csv is given, and the
       same resolution build_balanced_sample.py's DATASET_PATH uses.
    3. Falls back to the literal string 'first_turns.csv' if neither is
       found, so pd.read_csv() raises its own clear FileNotFoundError
       naming that path, instead of this function raising first.
    """
    local_candidate = "first_turns.csv"
    if os.path.exists(local_candidate):
        return local_candidate
    try:
        import importlib.resources

        packaged = importlib.resources.files("anthro_benchmark.prompt_sets") / "first_turns.csv"
        if packaged.is_file():
            return str(packaged)
    except Exception:
        pass
    return local_candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified subsampling for AnthroBench's first_turns.csv. "
        "Works unmodified on Colab, Kaggle, or a local checkout: run it from "
        "the directory containing first_turns.csv (or pass --input), and "
        "outputs are written to the current directory unless --output-dir "
        "says otherwise."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help=(
            "Path to first_turns.csv. If omitted: looks for "
            "'./first_turns.csv' in the current directory first, then falls "
            "back to the copy bundled in an installed anthro_benchmark "
            "package (requires `pip install -e .`)."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Directory to write output CSVs to. Defaults to the current "
        "working directory.",
    )
    parser.add_argument(
        "--k-per-cue",
        type=int,
        default=5,
        help="Prompts to keep per cue for the recommended block-stratified "
        "sample (max 10; default 5, i.e. 50%%).",
    )
    parser.add_argument(
        "--row-n",
        type=int,
        default=300,
        help="Target row count for the alternative row-stratified sample "
        "(default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for both sampling strategies (default: 42).",
    )
    parser.add_argument(
        "--skip-row-sample",
        action="store_true",
        help="Only run the recommended block-stratified sample; skip the "
        "row-stratified alternative.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    input_path = args.input or _default_input_path()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    validate_structure(df)
    print("Structural assumptions verified.\n")

    print(f"=== Recommended: block-stratified sample, {args.k_per_cue}/10 prompts per cue ===")
    sub = block_stratified_sample(df, k_per_cue=args.k_per_cue, random_state=args.seed)
    check_balance(df, sub)
    block_output_path = os.path.join(
        args.output_dir, f"first_turns_subset_balanced_k{args.k_per_cue}.csv"
    )
    sub.to_csv(block_output_path, index=False)
    print(f"Saved to: {block_output_path}")

    if not args.skip_row_sample:
        print(f"\n=== Alternative: row-stratified sample, n={args.row_n} ===")
        sub2 = row_stratified_sample(df, n=args.row_n, random_state=args.seed)
        check_balance(df, sub2)
        row_output_path = os.path.join(
            args.output_dir, f"first_turns_subset_row_n{args.row_n}.csv"
        )
        sub2.to_csv(row_output_path, index=False)
        print(f"Saved to: {row_output_path}")
