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
"""

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
    for key, k in alloc.items():
        if k <= 0:
            continue
        mask = np.ones(len(df), dtype=bool)
        keys = key if isinstance(key, tuple) else (key,)
        for col, val in zip(strata_cols, keys):
            mask &= (df[col] == val)
        parts.append(df[mask].sample(n=int(k), random_state=random_state))
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


if __name__ == '__main__':
    df = pd.read_csv('/mnt/user-data/uploads/first_turns.csv')
    validate_structure(df)
    print("Structural assumptions verified.\n")

    print("=== Recommended: block-stratified sample, 50% of prompts per cue ===")
    sub = block_stratified_sample(df, k_per_cue=5, random_state=42)
    check_balance(df, sub)
    sub.to_csv('/mnt/user-data/outputs/first_turns_subset_balanced_50pct.csv', index=False)

    print("\n=== Alternative: row-stratified sample, arbitrary n=300 ===")
    sub2 = row_stratified_sample(df, n=300, random_state=42)
    check_balance(df, sub2)
