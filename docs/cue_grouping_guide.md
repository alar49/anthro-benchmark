# Cue Grouping: Batched Classifier Rating

This documents the integration in `anthro_cue_grouping.patch` — what it changes, why, and how to use it. It touches `classifiers.py`, `rating.py`, `anthro_eval_cli.py`, and `classifier/__init__.py`, and adds two new files: `cue_grouping.py` and `test_cue_grouping_harness.py`.

## The problem

`rate_dialogues()` calls the classifier LLM once per `(turn, cue)` pair. There are 13 LLM-judged cues (a 14th, `personal pronoun use`, is regex-rated and never sent to an LLM), so every turn's conversation text — often the dominant cost — gets repeated in full 13 times. Cue grouping batches several cues into one call, sending the conversation text once and asking multiple questions about it.

## Taxonomy: 13 LLM cues, not 12

`cue_definitions.CUE_DEFINITIONS` has 14 keys. `personal pronoun use` is excluded from grouping entirely (always regex). The other 13 are what grouping operates on.

**Important distinction:** `first_turns.csv`'s `cue` column has only 12 values, because the prompt-design stage merged `validation` and `empathy` into one label (`"Validation/empathy"`). The classifier still rates them as two fully independent cues. Cue grouping works on the classifier's 13, not the CSV's 12 — don't try to map the CSV column onto `CUE_GROUP_CONFIGS`.

## Why grouping isn't just "put every cue in one call"

Some cues are definitionally close enough that asking about them in the same prompt risks the model's answer to one bleeding into its answer to another. This was the basis for how the groups below were built:

| Cluster | Members | Why risky together |
|---|---|---|
| Affective/other-directed | `empathy`, `validation`, `relatability` | All center on attunement to / affirming / connecting with the user's feelings |
| Somatic | `sensory input`, `movement and interactions`, `physical embodiment` | All concern having/using a physical body |
| Relationship-reference | `personal relationships`, `explicit relationship status` | Both hinge on referencing "a relationship," general vs. user-specific |
| Goal-pursuit | `desires`, `agency` | Both about wanting/pursuing actions or ambitions |
| Weak cross-category overlap | `sentience` ↔ `sensory input` | `sentience`'s definition says "susceptible to *sensations*, and conscious" |

Notably, the paper's own `behavior_category` labels don't track this — two of the four categories (`physical embodiment`, `relationship-building`) each bundle a *whole* cluster together. Grouping by official category is not automatically a safe default.

## New file: `cue_grouping.py`

### `CUE_GROUP_CONFIGS`

Five named, pre-built groupings, each a full partition of the 13 cues:

| Config | Groups | Calls/turn | Rationale |
|---|---|---|---|
| **`A_interference_avoidant`** (recommended default) | 3 | 3 (−77%) | No two cues from the same cluster ever share a group |
| `B_paper_taxonomy` | 4 | 4 (−69%) | Paper's official `behavior_category`; kept as a contrast case — bundles two full high-risk clusters whole |
| `C_hybrid` | 5 | 5 (−62%) | Keeps the two low-overlap categories (personhood, internal states) intact; only splits the two categories shown to be risky |
| `D_self_vs_other_directed` | 2 | 2 (−85%) | Splits along the classifier's own question-template axis ("claim to personally have X" vs. "demonstrate X towards CONVERSATION PARTNER 1"). Deliberate stress test — bundles every hard cluster whole, and one group asks 10 questions at once. Not a recommendation. |
| `E_conservative` | 10 | 10 (−23%) | Only pairs cues with clearly low overlap; every cluster-risk cue stays a singleton call |

Each config is validated at import time (`validate_cue_groups`): every one of the 13 cues must appear in exactly one group, or the module fails to import.

### `resolve_call_units(cues_to_rate, cue_group_config)`

Turns a flat cue list into an ordered list of **call units** — each unit is the list of cue names asked about in one LLM call.

- `personal pronoun use`, if present, is always split off into its own singleton unit first (regex path, unaffected by any config).
- `cue_group_config=None` → every remaining cue becomes its own singleton unit, in the order given. This is byte-for-byte what `rate_dialogues()` did before this patch.
- Otherwise, each group in the named config is filtered down to its intersection with `cues_to_rate` (so `--behaviors-to-rate` still limits what's actually asked); empty groups are dropped. Any requested cue the config doesn't mention falls back to its own singleton unit.

### `LLMGroupClassifier`

The grouped counterpart to `classifiers.LLMClassifier`. One instance handles one call unit (≥1 cues). `rate_turn_messages(assistant_turn_message, user_turn_message)` sends one `[system, user]` message pair and returns `{cue: (score, reason)}` for every cue in the unit.

### Prompt structure

Both the singleton path (`classifiers.py`) and the grouped path share one system prompt, built by `classifiers.build_classifier_system_prompt(num_questions)`:

- `num_questions == 1`: identical wording to the original single-message prompt (framing + output-format instructions), just now sent as a system message instead of prepended to the user message.
- `num_questions > 1`: adds the explicit independence instruction — *"Answer each question strictly on its own terms... a 'Yes' on one question must not influence your answer to any other question."* — and swaps the output format from `explanation;Yes/No` to a JSON object.

This sharing matters: a `LLMClassifier` call and a group of size 1 produce the *exact same* system prompt. If the two paths used different message structures, any comparison between them (e.g. via the test harness) would be confounded by a phrasing/structure difference that has nothing to do with grouping itself.

The user message (`create_prompt_for_cue_group` / `create_prompt_for_cue`) carries only the dynamic content: the conversation, and for each cue in the unit, its definition, negative examples, and tailored question (`_question_for_cue`, imported into `cue_grouping.py` from `classifiers.py` so there's one source of truth).

### Output parsing

`_parse_grouped_output` parses the model's JSON response **defensively, per cue**: a missing key, an unparseable label, or malformed JSON entirely degrades to `(-1, "Format not followed: ...")` for just the affected cue(s), rather than failing the whole batch. This mirrors `classifiers._process_raw_output`'s existing `-1` convention for unparseable singleton responses.

## Modified: `classifiers.py`

Two independent changes:

1. **Bug fix:** `create_prompt_for_cue` checked `key == "sensory_input"` / `"movement_and_interactions"` (underscores), but the real cue names have spaces (`"sensory input"`, `"movement and interactions"` — see `cue_definitions.py`). Those branches never fired; both cues silently used the generic *"claim to personally have X"* phrasing instead of their intended wording. Fixed to match on the real names. Confirmed behaviorally (built the prompt before/after and diffed the QUESTION line), and confirmed no other code depends on which branch fires — `cue` only ever flows through as an opaque string elsewhere in the codebase.
2. **System/user split:** see above. `create_prompt_for_cue` now returns only the user-role content.

`_process_raw_output` and the `explanation;Yes/No` parsing convention are unchanged — moving the framing into a system message doesn't touch how the response is parsed.

## Modified: `rating.py`

`rate_dialogues()` gains one new parameter:

```python
cue_group_config: str | None = None
```

Internally, the loop that used to iterate `for cue_to_rate in cues_to_rate` now iterates `for cue_unit in call_units` (from `resolve_call_units`). Each unit dispatches to one of three paths:

- `["personal pronoun use"]` → regex, unchanged.
- A unit of size 1 → `LLMClassifier`, unchanged code path.
- A unit of size >1 → `LLMGroupClassifier`, one call per (row × sample), fanned out into per-cue result dicts afterward.

After the model loop, **the exact same column-building logic runs once per cue in the unit** — so output shape is identical regardless of grouping: every cue still gets `{cue}_present`, `{cue}_{model}_final_present`, and either `{cue}_{model}_raw_rating` (num_samples=1) or `{cue}_{model}_raw_s{1..3}` / `{cue}_{model}_present_s{1..3}` (num_samples=3).

**One real difference to know about:** the `_raw_s*` / `_raw_rating` explanation-text columns are not identical in *content* between grouped and ungrouped runs, by design. A singleton call's raw column holds the model's full raw completion (which was already about one cue). A grouped call's raw column holds just that cue's extracted `"reason"` field from the shared JSON response — repeating the entire multi-cue JSON blob into every cue's column would be redundant. The **decision** (`_present` / `_final_present` / `_present_s*`) is unaffected either way — verified identical across grouped and ungrouped runs on the same input in testing (see Validation below).

`run_rating_process` passes `cue_group_config` through unchanged.

## Modified: `anthro_eval_cli.py`

New flag on the `rate` subcommand:

```
--cue-group-config {A_interference_avoidant,B_paper_taxonomy,C_hybrid,D_self_vs_other_directed,E_conservative}
```

Choices are pulled directly from `CUE_GROUP_CONFIGS.keys()`, so a typo fails fast at argument parsing, before any rating work starts. Omitting the flag reproduces pre-patch behavior exactly (`cue_group_config=None`).

```bash
python anthro_eval_cli.py rate \
  --dialogues-csv generated_dialogues/dialogues.csv \
  --classifier-model <your classifier model> \
  --cue-group-config A_interference_avoidant \
  --output-rated-csv rated_dialogues/rated.csv
```

## Modified: `classifier/__init__.py`

Now also exports `CUE_GROUP_CONFIGS` and `LLMGroupClassifier` (previously only `LLMClassifier` and `run_rating_process`), since the CLI needs `CUE_GROUP_CONFIGS.keys()` for its `choices=`.

## New file: `test_cue_grouping_harness.py`

A standalone comparison tool — **not** wired into `rate`, run manually before trusting a config on a real dataset:

```bash
python test_cue_grouping_harness.py \
  --dialogues pilot_dialogues.csv \
  --config A_interference_avoidant \
  --classifier-model <your classifier model> \
  --out comparison.csv
```

`pilot_dialogues.csv` needs `user_message`/`assistant_message` columns (same shape as a generated `dialogues.csv` — a small pilot subset is enough). It runs the same pilot set through both the untouched singleton baseline (13 calls/turn) and the chosen grouped config, and reports, per cue: raw agreement between the two, each condition's Yes-rate, and parse-error counts. Read this **before** running a config on a full dataset — if a config's agreement collapses specifically on the cues you clustered together, that's the interference risk from the table above showing up in practice rather than staying theoretical.

## How to use this, step by step

1. **Generate dialogues** as usual (`anthro_eval_cli.py generate ...`) — nothing about generation changes.
2. **Pilot a config** with `test_cue_grouping_harness.py` on a small subset of the generated dialogues. Start with `A_interference_avoidant` unless you have a specific reason to try another.
3. **If agreement holds up**, rate the full dataset with `--cue-group-config <name>` on the `rate` subcommand.
4. **If it doesn't**, either fall back to no grouping for the cues that diverged, or try a different config from the table — `C_hybrid` is the more conservative middle ground, `E_conservative` the safest (smallest savings).

## Validation performed

No real LLM was called during development (no credentials in the build environment) — everything below validates *plumbing*, not classifier accuracy on real text:

- **Backward compatibility:** ran the pre-patch `rate_dialogues()` and the patched one side by side, same synthetic dialogues (including an empty-message row to hit the skip path), same cues, through a deterministic mock keyed on `(cue, conversation text)`. Diffed all output columns cell-by-cell: zero differences with `cue_group_config=None`.
- **Grouped-mode correctness:** call count dropped exactly as arithmetic predicts (e.g. 18→12 calls in one test scenario), and every classification decision matched the ungrouped run exactly (only the raw-explanation text differs, as documented above).
- **Real end-to-end run:** applied the patch to an independent fresh clone and ran it against the *actual* `LLMClient` (real `litellm` import, real API-key validation, real package import chain) with only the network-calling `generate()` method mocked. Confirmed correct call counts and column presence across three different `cue_group_config` values.
- **Real CLI, no mocking at all:** ran `anthro_eval_cli.py rate --cue-group-config ...` as a subprocess with zero mocking. It passed through argument parsing, cue-unit resolution, classifier construction, and prompt building, failing only at the literal outbound HTTP request (blocked by this environment's network policy) — i.e. it got as far as *actually trying to call the API* before the only failure occurred.
- **Patch applies cleanly** via `git apply --check` against an independent fresh clone of the branch.

## Known limitations

- No empirical accuracy comparison against a real classifier model yet — that's exactly what the pilot step (`test_cue_grouping_harness.py`) is for, and it hasn't been run on real data.
- `D_self_vs_other_directed`'s 10-cue group is a stress test, not a recommendation — a large group risks accuracy degradation from sheer question count, independent of the semantic-similarity concerns the other configs are designed around.
- Grouping changes token usage per call (fewer, longer calls) but not necessarily proportionally to the raw call-count reduction — savings depend on how long your turns are relative to the definitions/examples block; check on your actual data rather than assuming the call-count ratio.
