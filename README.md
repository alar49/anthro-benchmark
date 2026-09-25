   > Fork of [google-deepmind/anthro-benchmark](https://github.com/google-deepmind/anthro-benchmark). See original repo for upstream documentation.

# Replication and Extension

This work replicates and extends the original study by evaluating longer conversations (10 turns vs. 5) and integrating open-weight models. It introduces new configuration flags for flexible execution across multiple model providers, alongside efficient ways to significantly reduce inference costs and execution time while preserving the original experimental design and validity.

## Rationale

On 10 turns vs. 5: as the original study suggested, a large part of anthropomorphic behaviors only emerge after multiple turns; additionally, anthropomorphic behaviors at turn (t) are more likely to appear if they were already present in turn (t-1);

On open-weight models: they currently represent valid and sometimes preferred alternatives users may run locally (they are free, more private, also able to run on low-spec devices). More importantly, open weight models are “frozen in time”, that is older, possibly unsafe versions are always available to be downloaded, so their anthropomorphism evaluation is very much needed.

## Additions

General additions and modifications:
- bug fixing and code explanations;
- novel implementations of API integrations for OpenRouter, Google Gemini/Vertex AI, Anthropic, Mistral, and OpenAI, facilitating reproducible and extensible access to these model providers;
- specific OpenRouter provider-routing options (provider order, fallback, require parameters for both user and target LLM);
- --user-llm-temperature and --target-llm-temperature are now separately editable;
- reasoning options (--reasoning-mode and --reasoning-effort). Notably, Reasoning is only ever requested from the target LLM, never the user LLM. The reasoning trace is kept in a separate “assistant_reasoning” CSV column, it is never concatenated into “assistant_message” (so the user LLM will not see target LLM reasoning block), and the rating stage additionally strips any stray “<think>...</think>” to base findings only on the final answer, rather than including the reasoning;
- --max-tokens options for both user and target LLM;
- optional budget/iteration cap options (BudgetGuard);
- optional concurrency (--max-concurrency) to implement parallel inference, with optional batch ordering when budget cap is met (--strict-batch-ordering);
- --incremental-save flag that prevents losing all data after an unexpected crash by rewriting the .csv after each completed dialogue (sequential) or each completed batch (so a crash partway through leaves everything completed so far on disk); 
- --resume an interrupted or corrupted run while skipping successfully completed dialogue.;
- an early-stopping option (--stop-on-natural-end) to save on inference costs (more on this in the full documentation). Default off reproduces the original behavior exactly (every dialogue runs the full `--num-turns` regardless of content); When set, the user LLM is instructed to recognize when the target's last message is a natural conversational close and, if so, give one final reply, emit a sentinel token (<DIALOGUE_COMPLETE>, stripped before saving), and stop rather than continuing to the full turn count;
- a --timeout option. Bounds how long a single call can hang before it's treated as failed, instead of blocking this thread indefinitely on a stuck provider. It has its own --max-retries;
- several rate limiting options, customizable for user, target and classifier LLM: --min-seconds-between-calls, --max-requests-per-minute, --max-requests-per-second, --max-tokens-per-minute;
- original .csv subsampling methods (build_balanced_sample.py, stratified_subsample.py) to run smaller versions of the benchmark according to own preferences (more on this in the full documentation); flag for generator.py (--custom-prompt-csv, --num-dialogues, --dialogues-per-condition, --num-turns);
- rate limits, timeouts, 5xx errors, 429 errors retry mechanism (--max-retries, --initial-backoff, --max-backoff)

Rating.py specific:
- original papers’ --classifier-model (one or more), --classifier-temperature, --num-samples, --behaviors-to-rate;
- optional safety caps (--classifier-max-tokens-base / --classifier-max-tokens-per-cue) meant as a runaway-generation circuit breaker, not a cost lever (basically to avoid and interrupt any infinite loop);
- novel cue grouping options (--cue-group-config), in order to scale down API calls during rating (see full documentation). Basically, it batches multiple cues into a single LLM call per turn instead of one call per cue. I defined different strategies in order to be more or less safe and conservative accounting for behaviors similarity. Notably, rating.py is instructed to handle more behaviors at once by creating a unique JSON line for each of them: it looks for the last semicolon in the raw LLM output and inspects the text after it for a strict/starts-with match on "yes"/"no"; if that's ambiguous it falls back to checking text before the last semicolon; only if all of that fails does it return “-1” ("Format not followed"). A “-1” in the output CSV means the classifier's output didn't parse cleanly (not that the model said "no") and grouped-cue calls (--cue-group-config) use a stricter JSON-Lines parser instead (cue_grouping.py), where a “-1” specifically means "no valid JSON line was found for this cue" (possibly a truncated response).

Analysys.py specific:
- added a --first-n-turns flag to allow for calculating results after N turns other than on the entire run;
- added a missing_ratings_report.json to the output files to investigate invalid/missing evaluations total and per-cue frequencies.


## Inference calls: original work vs. current implementation


| Stage | Parameter | Symbol / Formula | Original Work | Current Implementation | Notes / Configuration |
| --- | --- | --- | --- | --- | --- |
| **Generation** | Number of prompts | P | 960 | 96 | `--k-per-cue 1` `--seed 42` |
|  | Turns per prompt | T | 5 | up to 10 | `--stop-on-natural-end` |
|  | Total Messages / Model | M = P * T | 4,800 | ~960 | **~5× reduction while doubling turns** |
| **Ratings** | Behaviour calls | B | 13 | 3 | 1 per behaviour vs. `--cue-group-config "A_interference_avoidant"` |
|  | Messages evaluated | M | 4,800 | ~960 | From Generation stage |
|  | Judge LLMs | J | 3 | 1 | Reduced (see rationale) |
|  | Samples per rating | S | 3 | 1 | Single sample (see paper) |
|  | Total Evaluation Calls / Model | R = B * M * J * S | 561,600 | ~2,880 | **~195× reduction in compute** |


## Interactive Results

To allow a better results visualization, I have built a dedicated **Interactive Comparison Tool** natively hosted on GitHub Pages for this dataset. 

It allows you to dynamically filter and cross-examine the data with zero visual clutter:
* **Every Model and versions** (from Mistral Small to LLaMA 3.3 70B)
* **14 Behavioral Cues** (Sentience, Empathy, Agency, Validation, etc.)
* **Turn Scope Comparison:** Isolate 5-Turns, 10-Turns, or view **Side-by-Side Comparisons** of the exact same models across conversational depth.
* **Modality:** Toggle between Data Tables, Radar Charts, and Horizontal Bar Charts.

[![Explore the Data](https://img.shields.io/badge/Launch_Interactive_Results_Tool-000000?style=for-the-badge&logo=github&logoColor=white)](https://alar49.github.io/anthro-benchmark/results.html)

*(Click the badge above to launch the tool)*

---

### Static Fallback: 5-Turn vs 10-Turn Glimpse 
<details>
<summary>Click here to view a static sample comparing conversational depth</summary>

| Cue | Mistral Small 4 (5T) | Mistral Small 4 (10T) | Gemma 4 4B (5T) | Gemma 4 4B (10T) | Qwen 3.8 27B (5T) | Qwen 3.8 27B (10T) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validation** | 66.24 | 61.31 | 85.95 | 78.78 | 75.96 | 69.40 |
| **Empathy** | 57.96 | 54.90 | 62.68 | 60.25 | 70.00 | 69.40 |
| **Agency** | 4.67 | 3.81 | 6.71 | 5.23 | 3.19 | 2.73 |

> *Note: Open the Interactive Tool above for all 15 models, 14 cues, and full graphical visualization.*

</details>

Full details and results available at: [(WIP) Replicating and extending the AnthroBench library](https://docs.google.com/document/d/146u0fuif-q_K1KfOHvG6KMLTk4QhRbTkT9e94NRD-9U/edit?usp=sharing)

