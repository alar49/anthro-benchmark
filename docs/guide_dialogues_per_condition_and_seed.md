# Guide: `--dialogues-per-condition` and per-run sampling seeds

Both features below were implemented and executed against synthetic data to
verify their behavior (not just syntax-checked) — see "What was tested" at
the end of each section for exactly what was run.

> This guide covers `build_balanced_sample.py` specifically. For how it
> compares to `stratified_subsample.py`'s two sampling functions (different
> sampling unit, different trade-offs, and a seed bug found and fixed in
> `row_stratified_sample`), see `docs/sampling_strategies_guide.md`.

---

## 1. Controlling dialogues per condition

### What changed

`anthro_eval_cli.py generate` now accepts `--dialogues-per-condition` as an
alternative to `--num-dialogues`. It is **mutually exclusive** with
`--num-dialogues` — passing both raises an error immediately (at the
argparse level, and defensively again inside `DialogueGenerator` if you call
it directly from Python).

```bash
anthro_eval_cli.py generate \
  --user-llm-model <model> \
  --target-llm-model <model> \
  --custom-prompt-csv balanced_first_turns_sample_seed156.csv \
  --dialogues-per-condition 2
```

### The math

A "condition" is a unique combination of whichever of these columns are
present in the loaded prompt data: `use_domain`, `use_scenario`, `empathy`,
`professionalism`, `cue`, `behavior_category` (renamed to `category` by
`_load_prompts`). This is exactly `CONDITION_COLUMNS` from
`build_balanced_sample.py`.

```
total dialogues generated = dialogues_per_condition × (number of unique conditions found)
```

For your own example: 96 conditions × `--dialogues-per-condition 2` → 192
dialogues, 2 per condition.

**This holds regardless of how many prompt-variant rows each condition
already has in the CSV.** If a condition has fewer rows than
`dialogues_per_condition`, rows are cycled (reused) to make up the count; a
console message tells you when this is happening, e.g.:

```
Note: every condition has 1 row(s) available, so with dialogues_per_condition=2
rows will be reused (cycled) to reach the requested count.
```

This means the two scripts are now decoupled on purpose:

| Script | Controls |
|---|---|
| `build_balanced_sample.py` (`number_of_entries_per_combination`) | how many *distinct prompt phrasings* are sampled per condition |
| `--dialogues-per-condition` on the generator | how many *dialogues* get generated per condition, regardless of how many phrasings are available |

If your balanced sample has 1 row per condition (the simplest setup —
`number_of_entries_per_combination=1`) and you ask for
`--dialogues-per-condition 3`, you get 3 replicate dialogues per condition,
all starting from the same prompt phrasing, differing only in what the two
LLMs generate stochastically. If your sample already has 2 rows per
condition and you ask for `--dialogues-per-condition 2`, each of the 2
phrasings gets used exactly once — no reuse needed.

### Requirements and failure modes

- Requires the loaded prompt data to contain **at least one** of the
  recognized condition columns. A fully custom CSV with only a `prompt`
  column (no domain/scenario/cue/category metadata) will raise a clear
  `ValueError` — use `--num-dialogues` for that case instead, since "per
  condition" has no defined meaning there.
- Passing both `--num-dialogues` and `--dialogues-per-condition` raises an
  error rather than silently picking one.
- If conditions in your CSV have **uneven** row counts (some have 1 row,
  others have 3, say), you still get exactly `dialogues_per_condition`
  dialogues for every condition — the console prints a warning so you're
  aware some conditions are reusing rows more than others, rather than this
  passing silently.

### A caveat worth knowing before you rely on this for analysis

The saved `dialogues.csv` output only carries `prompt_category`,
`prompt_cue`, `prompt_use_domain`, and `prompt_use_scenario` in its metadata
columns — **not** `empathy` or `professionalism`, even though those are two
of the six columns that define a "condition" for stratification and for
`--dialogues-per-condition`'s grouping. This is a pre-existing trait of
`_generate_single_dialogue`'s metadata dict, not something introduced by
this change, but it means that if your conditions are distinguished by
empathy/professionalism (which `CONDITION_COLUMNS` says they are), you
cannot fully reconstruct which condition a given output row belongs to from
the dialogues CSV alone — you'd need to join back to the source prompt CSV
(e.g., on `prompt_text`, or by adding a row ID). Flagging this now since it's
a real gap between what's sampled and what's saved, not because it needed
fixing for this specific request.

### What was tested

Built a stub `anthro_benchmark` package (fake `LLMClient`/`Role`/
`BudgetGuard`, no real API calls) and ran `DialogueGenerator` and the actual
CLI argument parser end-to-end against synthetic CSVs, confirming:
- 1 row/condition → exact per-condition counts, no reuse.
- 2 rows/condition with `dialogues_per_condition=3` → correct reuse pattern.
- Uneven group sizes (1 vs 2 rows) → still exactly N per condition.
- `ValueError` on: both flags set together, and a CSV with no condition
  columns.
- The CLI's `--num-dialogues | --dialogues-per-condition` mutually exclusive
  group rejects both being passed, with a normal argparse usage error.
- The seed-embedded sample CSV from part 2 below, fed into
  `--dialogues-per-condition`, produces exactly the expected counts.

---

## 2. Per-run random seed for `build_balanced_sample.py`

### What changed

The hardcoded `RANDOM_SEED = 42` is gone. The script now:

- Draws a fresh seed from `[0, 999]` **every time you run it**, unless you
  override it.
- Prints the seed used, and the exact flag to reproduce that run.
- Embeds the seed in the output filename: `balanced_first_turns_sample_seed156.csv`.
- Adds a `sampling_seed` column to the output CSV itself, so the seed
  survives even if the file gets renamed or copied later.

```bash
# Run 1: no seed given -> a new random one is drawn and reported
$ python build_balanced_sample.py
Using seed 156 (drawn at random from [0, 999]).
To reproduce this exact sample later, re-run with: --seed 156
...
Sampled dataset saved to: balanced_first_turns_sample_seed156.csv

# Run 2: no seed given again -> a DIFFERENT random sample
$ python build_balanced_sample.py
Using seed 571 (drawn at random from [0, 999]).
...
Sampled dataset saved to: balanced_first_turns_sample_seed571.csv

# Reproduce Run 1 exactly, any time later:
$ python build_balanced_sample.py --seed 156
Using seed 156 (explicitly provided via --seed).
...
Sampled dataset saved to: balanced_first_turns_sample_seed156.csv   # byte-identical to Run 1
```

This is the actual point of your idea: a single fixed seed gives you
determinism but zero variation across runs (every run is identical
forever); a per-run random seed gives you variation, and recording it (in
both the filename and a data column) is what lets you deterministically
redo any *specific* one of those runs on demand.

### Notes / things not changed

- `number_of_entries_per_combination`, `ALLOW_REPLACEMENT`, `DATASET_PATH`,
  and `CONDITION_COLUMNS` are still plain module-level constants, not CLI
  flags — only `--seed` was added, since that's what you asked about. Let me
  know if you also want CLI flags for those; it's a small extension of the
  same `argparse` setup.
- The auto-drawn range is `[0, 999]` (hence the 3-digit `seedXXX` filename
  suffix). If you pass `--seed` explicitly with a value outside that range
  (e.g. `--seed 12345`), it's used as-is and not validated — only the
  *automatic* draw is bounded.
- The per-group derived seeds inside the sampling loop
  (`group_seed = seed + group_position`) and the final cross-combination
  shuffle both now use this resolved `seed` instead of the old fixed
  constant; nothing else about that logic changed.

### What was tested

Ran the actual edited script (not a mock) against a synthetic 8-condition,
5-variant-per-condition dataset:
- Two runs with no `--seed`: got two different seeds (156, 571) and
  confirmed the two output DataFrames were **not** identical.
- Two runs with `--seed 156`: confirmed the output DataFrame was **byte-for-byte
  identical** to the original run 156 output loaded back from disk, and to
  each other.
- Confirmed `sampling_seed` column and `_seed156`/`_seed571` filename
  suffixes appear correctly.
- Confirmed `--help` output renders correctly.

---

## Files changed

- `anthro_benchmark/generator/generator.py` — new `dialogues_per_condition`
  parameter, condition-grouping helper, condition-aware prompt selection.
- `anthro_eval_cli.py` — new `--dialogues-per-condition` flag (mutually
  exclusive with `--num-dialogues`), wired through to `DialogueGenerator`,
  reflected in the output CSV filename (`_dpcN`).
- `build_balanced_sample.py` — new `--seed` flag, random-by-default seed
  resolution, seed embedded in output filename and as a CSV column.

A unified diff of all three (`goal2_dialogues_per_condition_and_seed_changes.patch`)
is included alongside the full updated files, matching the style of your
existing `goal1_cli_generator_changes.patch`.
