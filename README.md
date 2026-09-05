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



Full details and results available at: [WIP]



# AnthroBench

A library for generating, rating, and analyzing dialogues to evaluate anthropomorphic behaviors in LLMs, developed in [AnthroBench: A Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models](https://arxiv.org/abs/2502.07077).

## Structure

The library is organized into several key packages and modules:

- `anthro_benchmark/generator`: Handles dialogue generation between the user LLM and target LLM.
- `anthro_benchmark/classifier`: Contains logic for classifying dialogue turns based on anthropomorphic behaviors, including the `LLMClassifier` and the `cue_definitions.py` behavior definitions.
- `anthro_benchmark/core`: Core utilities, including `llm_client.py` for interacting with various LLM APIs.
- `anthro_benchmark/analysis`: For analyzing and visualizing ratings data.
- `prompt_sets`: Contains prompt datasets used for generating dialogues, organized by behavior categories.
- `anthro_eval_cli.py`: The command-line interface script.
- `setup.py`: For package installation and distribution.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/google-deepmind/anthro-benchmark.git
    cd anthro-benchmark
    ```

2.  Create and activate a Python virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    ```

3.  Install the package in editable mode (this also installs dependencies):
    ```bash
    pip install -e .
    ```

## API keys setup

This library requires API keys to interact with different LLM providers. You need to set up your keys as environment variables:

```bash
# OpenAI API key 
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic API key 
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google API key 
export GOOGLE_API_KEY="your-google-api-key"

# Mistral API key
export MISTRAL_API_KEY="your-mistral-api-key"
```

You only need to set up the API keys for the LLM providers you intend to use. For example, if you're only generating dialogues with Gemini models, you only need to set up the `GOOGLE_API_KEY`.

## Prompt sets

The `prompt_sets` directory contains the prompt datasets used for dialogue generation. The primary file is:

- `first_turns.csv`: The main dataset containing all prompts. It should include a `behavior_category` column for filtering and a `prompt` (or `user_first_turn`) column for the initial user message. Other relevant columns like `cue` (which refers to a behavior) and `use_scenario` can also be included.

### Behavior categories

Prompts in `first_turns.csv` can be organized by a `behavior_category` column. There are four categories available:

- `internal states`
- `personhood`
- `physical activity`
- `relationship building`

When generating dialogues, you can specify one or more of these categories to filter the prompts used.

## Command-line interface (`anthro-eval`)

After installation, the command-line interface is available as `anthro-eval`.

### 1. Generating dialogues

Generate dialogues using prompts filtered by behavior categories:

```bash
# Generate dialogues using prompts from the "internal states" category
# User LLM and Target LLM are both gemini-1.5-flash
anthro-eval generate --user-llm-model "gemini/gemini-1.5-flash" --target-llm-model "gemini/gemini-1.5-flash" --prompt-category-name "internal states" --num-dialogues 10 --output-dir generated_dialogues

# Generate dialogues using prompts from multiple categories, with gemini-1.0-pro as the target
anthro-eval generate --user-llm-model "gemini/gemini-1.5-flash" --target-llm-model "gemini/gemini-1.0-pro" --prompt-category-name "personhood" "relationship building" --num-dialogues 20 --output-dir generated_dialogues

# Generate dialogues filtering for specific behaviors within categories
anthro-eval generate --user-llm-model "gemini/gemini-1.5-flash" --target-llm-model "gemini/gemini-1.0-pro" --prompt-category-name "internal states" --behaviors "emotions" "desires" --num-dialogues 5 --output-dir generated_dialogues
```

The system loads prompts from `prompt_sets/first_turns.csv` and filters them based on the specified `--prompt-category-name`.

### 2. Rating dialogues

Rate generated dialogues for anthropomorphic behaviors. Behaviors are defined in `anthro_benchmark/classifier/cue_definitions.py`.

```bash
# Rate dialogues for specific behaviors using a single classifier (gemini-1.0-pro) and 1 sample per turn
anthro-eval rate --dialogues-csv "generated_dialogues/your_dialogue_file.csv" --classifier-model "gemini/gemini-1.0-pro" --behaviors-to-rate "empathy" "desires" --num-samples 1

# Rate dialogues using multiple classifier models (gemini-1.0-pro and gemini-1.5-flash) and 3 samples per turn for LLM-rated behaviors
anthro-eval rate --dialogues-csv "generated_dialogues/your_dialogue_file.csv" --classifier-model "gemini/gemini-1.5-pro" "gemini/gemini-1.5-flash" --behaviors-to-rate "empathy" "validation" --num-samples 3

# Rate dialogues for all available behaviors defined in cue_definitions.py using a single classifier
# This will include "first-person pronoun use" (rated by regex) if it's a key in cue_definitions.py
anthro-eval rate --dialogues-csv "generated_dialogues/your_dialogue_file.csv" --classifier-model "gemini/gemini-1.5-flash"
```

- You can specify one or more `--classifier-model` names. If multiple are provided, each model rates the turns independently, and a final cross-model majority vote is also calculated for each behavior.
- `--num-samples` can be `1` or `3`. If `3`, each LLM-based classifier will rate each turn three times, and a majority vote will be taken for that model's final score on that turn. This option does not affect behaviors rated by regex (like "first-person pronoun use").
- If `--behaviors-to-rate` is not specified, all behaviors from `cue_definitions.py` are rated.
- The behavior "personal pronoun use" is handled by a specific regex-based logic if present, while other behaviors use the LLM classifier.
- Rated dialogues are saved in the `rated_dialogues/` directory by default.

### 3. Analyzing results

Analyze the rated dialogues to generate summaries and plots:

```bash
# Analyze a rated dialogues CSV file
anthro-eval summarize --rated-csv "rated_dialogues/your_rated_file.csv" --output-dir analysis_results
```

Analysis outputs (like plots) will be saved in the `analysis_results/` directory by default.

## License

Apache-2.0 License
