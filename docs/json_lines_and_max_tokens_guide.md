# JSON Lines Rating Output & Classifier Token Limits

This documents `anthro_cue_grouping_v2.patch`, which changes how grouped classifier calls format their response and adds a `max_tokens` safety net. It builds on top of the cue-grouping integration from `anthro_cue_grouping.patch` — read that guide first if you haven't; this one assumes `CUE_GROUP_CONFIGS`, `resolve_call_units`, and `LLMGroupClassifier` already exist.

## 1. Why JSON Lines, not one JSON object

The original grouped-call format asked the model for a single JSON object nesting every cue in the call:

```json
{"sentience": {"reason": "...", "label": "No"}, "empathy": {"reason": "...", "label": "Yes"}}
```

The problem: `json.loads()` needs the *entire* response to be well-formed. If a `max_tokens` cutoff (or any other truncation) lands anywhere in that response — even after several cues have already been fully and correctly answered — the whole object fails to parse, and **every cue in that call comes back `-1`**, including ones the model had already finished answering before the cutoff.

JSON Lines fixes this by making each cue's answer a separate, independently-parseable unit:

```json
{"cue": "sentience", "reason": "...", "label": "No"}
{"cue": "empathy", "reason": "...", "label": "Yes"}
```

One line per cue, in the same order the questions were asked, nothing else in the response (no fences, no surrounding array, no numbering). `_parse_grouped_output` in `cue_grouping.py` now:

- Splits the response by line, extracts the outermost `{...}` span per line (tolerating stray numbering or prefixes), and parses each independently. A malformed or missing line only costs that one cue.
- Matches lines to cues by an explicit `"cue"` field, not by position — so out-of-order lines, or a model that slightly varies a cue name's case (`"Sentience"` vs `"sentience"`), still resolve correctly via a case-insensitive lookup.
- For any cue whose line never showed up (truncated before reaching it, or genuinely malformed), returns `(-1, "Format not followed: no line found for cue '<cue>' (response may have been truncated or malformed)...")` — same `-1` convention as everywhere else in the pipeline, but now scoped to just that cue.

**Verified directly**, not just by inspection: fed the parser a response missing its last line entirely (simulating a cutoff between cues) and one with a genuinely truncated last line (`"reason": "the model was cut off mid`) — in both cases every complete line before the cutoff parsed correctly, and only the affected cue came back `-1`.

### Consistent raw-text formatting

Each successfully-parsed cue's stored explanation is now `"{reason}; {Yes/No}"` (label normalized to capitalized `Yes`/`No` regardless of how the model cased it), matching `classifiers.LLMClassifier`'s raw-output convention exactly. Before this patch, a grouped call's raw column held only the reason text with no trailing verdict, while a singleton call's held the full `"reason; Yes"` string — cosmetically inconsistent for anyone reading the CSV by eye, even though nothing downstream (`analysis.py` included) ever parsed that column programmatically. Now both look the same regardless of which path produced them.

### The size-1 group bug this also fixed

`LLMGroupClassifier` is used for any group, including groups of size 1 (several exist in `CUE_GROUP_CONFIGS["E_conservative"]`, e.g. `["sentience"]`). The system prompt's output-format instructions used to be selected by `num_questions > 1`, so a size-1 group would get the *singleton's* plain-text instructions while the parser still expected JSON — silently failing every such call. Fixed by making the format an independent parameter (`structured_output`) from the question count: `classifiers.py`'s true singleton path always passes `structured_output=False`; `cue_grouping.py`'s grouped path always passes `structured_output=True`, even for a group of one. This didn't affect real `rate_dialogues()` runs (which already route size-1 units to the true singleton classifier instead), but it did affect `test_cue_grouping_harness.py`, which builds `LLMGroupClassifier` directly for every group in a config regardless of size.

## 2. The two `max_tokens` flags

```
--classifier-max-tokens-base INT
--classifier-max-tokens-per-cue INT
```

The classifier's `max_tokens` for a given call is:

```
max_tokens = classifier_max_tokens_base + classifier_max_tokens_per_cue × (cues actually asked in that call)
```

Computed **per call unit, at runtime** — not per config — using the actual `len(cue_unit)` from `resolve_call_units`. This matters because `--behaviors-to-rate` can filter a config's groups down: if you ask for only 2 of `CUE_GROUP_CONFIGS["A_interference_avoidant"]`'s 5-cue group, that call gets budget for 2 cues, not 5.

Both flags default to `None`. If neither is set, no `max_tokens` key is added to the classifier config at all — behavior is completely unchanged from before this patch (unbounded, exactly as it's always been). Passing only one of the two works fine (the other is treated as 0).

**This is a circuit breaker, not a cost-optimization lever.** The intent is to catch a genuinely pathological response (a repetition loop, a model that won't stop) before it burns thousands of tokens for no reason — not to squeeze down your per-call cost. Set it several times larger than whatever a normal response actually takes; if you don't know that number yet, run a small pilot with no cap, look at `usage.completion_tokens` on the responses, and size from there. A cap set anywhere near your model's *normal* output length risks doing the opposite of what you want: truncating a legitimate answer before its verdict is ever written down.

Example: rating with config A, budgeting 40 base tokens plus 60 per cue —

```bash
python anthro_eval_cli.py rate \
  --dialogues-csv generated_dialogues/dialogues.csv \
  --classifier-model <your classifier model> \
  --cue-group-config A_interference_avoidant \
  --classifier-max-tokens-base 40 \
  --classifier-max-tokens-per-cue 60 \
  --output-rated-csv rated_dialogues/rated.csv
```

Config A's largest unit has 5 cues → `40 + 60×5 = 340` tokens for that call; its smaller units get less.

## 3. How max_tokens behaves *given* JSON Lines specifically

This is the part worth understanding before you pick numbers, because JSON Lines changes the *shape* of the risk, not whether the risk exists.

**The blast radius is bounded, not eliminated.** A tight cap can still stop the model mid-response before it reaches every cue in a group — JSON Lines doesn't let the model say more than its token budget allows. What changes is what you lose when that happens: with the old single-object format, a cutoff anywhere cost you every cue in the call. With JSON Lines, it costs you only the cues whose lines hadn't been fully written yet — anything already complete before the cutoff is unaffected. That's the entire benefit; it doesn't buy you an unbounded budget, it buys you graceful degradation instead of total failure.

**The failure is systematic, not random — and it's worth knowing which direction.** Generation is sequential: the model writes line 1, then line 2, and so on, in the same order the questions were asked. If a group's budget is a little too tight, it isn't a coin-flip which cue gets cut off — it's reliably whichever cues come **last** in that group's question order. If you ever see the same cue (or the same tail of a group) coming back `-1: no line found` across many rows while the earlier cues in that same group are fine, that's a sizing problem with your `max_tokens` for that group, not a random glitch worth re-running and hoping goes away.

**Practical implication:** if you want to lean on a tighter cap for cost reasons despite the "safety net, not lever" framing above, JSON Lines makes that somewhat less dangerous than it would have been with the old format — you'd be predictably losing the last cue or two of your larger groups rather than losing entire calls outright. But "somewhat less dangerous" isn't "safe": a systematically-truncated tail still means those specific cues are quietly missing data across your whole run, in a way that's easy to miss if you're only looking at overall `-1` rates rather than per-cue ones. Worth checking `-1` rates broken out by cue position within each group, not just in aggregate, if you do decide to run tight.
