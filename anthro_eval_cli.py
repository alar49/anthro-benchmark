# Copyright 2025 The Anthropomorphism Benchmark Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import re
import traceback

from absl import app
from absl.flags import argparse_flags

from anthro_benchmark.generator import (
    DialogueGenerator,
    LLMGenerationError,
    DEFAULT_USER_SYSTEM_PROMPT,
)
from anthro_benchmark.classifier import run_rating_process, CUE_GROUP_CONFIGS
from anthro_benchmark.core.llm_client import BudgetGuard, RateLimiter, estimate_cost_from_usage


# helper function for file/column name sanitation
def sanitize_model_name(model_name: str) -> str:
    """Removes characters problematic for filenames/column names."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name)

### EDIT FROM HERE ###

# replace the helper block
def _reasoning_mode_and_effort_from_args(args):
    """
    Resolve --reasoning-mode/--reasoning-effort into the (reasoning_mode,
    reasoning_effort) pair LLMClient actually needs, WITHOUT collapsing
    "off" and "default" into the same value.

    reasoning_mode is tri-state (see LLMClient.generate()'s docstring):
      - True:  --reasoning-mode on (or an explicit --reasoning-effort with
               --reasoning-mode left at its default) -- reasoning
               explicitly requested.
      - False: --reasoning-mode off -- reasoning explicitly turned off
               (LLMClient sends reasoning={"enabled": False} for this,
               not just a bare omission -- several reasoning-capable
               models default to reasoning ON when the `reasoning` key is
               absent entirely, so "say nothing" and "say off" are NOT
               the same request).
      - None:  --reasoning-mode default with no --reasoning-effort --
               nothing is sent; the provider's own default for that model
               applies untouched.

    Raises ValueError for the contradictory --reasoning-mode off combined
    with an explicit --reasoning-effort, rather than silently letting one
    win -- that combination almost certainly means the flags don't say
    what the caller intended.
    """
    if args.reasoning_mode == "off":
        if args.reasoning_effort:
            raise ValueError(
                "Both --reasoning-mode off and --reasoning-effort "
                f"{args.reasoning_effort!r} were set -- these contradict "
                "each other (an effort level only makes sense when "
                "reasoning is on). Drop one."
            )
        return False, None

    if args.reasoning_effort:
        return True, args.reasoning_effort

    if args.reasoning_mode == "on":
        return True, "medium"  # OpenRouter default when reasoning is enabled

    return None, None  # "default": don't touch the `reasoning` key at all


def _reasoning_effort_from_args(args):
    """Back-compat shim: reasoning_effort half of
    _reasoning_mode_and_effort_from_args, for any caller that only needs
    that part."""
    return _reasoning_mode_and_effort_from_args(args)[1]


def _openrouter_provider_from_args(args, prefix: str) -> Optional[Dict[str, Any]]:
    """
    Assemble an OpenRouter `provider` routing object (see
    https://openrouter.ai/docs/guides/routing/provider-selection) from the
    --{prefix}-openrouter-* flags for one LLM role.

    `prefix` is "user_llm" or "target_llm", matching the argparse dest
    names generated from --user-llm-openrouter-* / --target-llm-openrouter-*.

    Returns None (leaving OpenRouter's own default load-balancing untouched)
    if none of the relevant flags were set for that role, so callers that
    don't care about this never see an empty dict cluttering their config.
    """
    order = getattr(args, f"{prefix}_openrouter_provider_order", None)
    no_fallbacks = getattr(args, f"{prefix}_openrouter_no_fallbacks", False)
    require_parameters = getattr(args, f"{prefix}_openrouter_require_parameters", False)

    provider: Dict[str, Any] = {}
    if order:
        provider["order"] = list(order)
    if no_fallbacks:
        provider["allow_fallbacks"] = False
    if require_parameters:
        provider["require_parameters"] = True

    if no_fallbacks and not order:
        print(
            f"Warning: --{prefix.replace('_', '-')}-openrouter-no-fallbacks was set "
            f"without --{prefix.replace('_', '-')}-openrouter-provider-order. This "
            "disables fallbacks for OpenRouter's own default (price-based) provider "
            "choice, rather than pinning to a specific provider list -- almost "
            "certainly not what you want."
        )

    return provider or None


def _build_budget_guard(args, session_id: str):
    max_iterations = getattr(args, "max_iterations", None)
    max_budget_per_session = getattr(args, "max_budget_per_session", None)

    if max_iterations is None and max_budget_per_session is None:
        return None

    if max_iterations is None:
        raise ValueError("--max-iterations is required when using budget controls.")

    cost_estimator = None
    if max_budget_per_session is not None:
        input_cost = getattr(args, "input_cost_per_1m_tokens", None)
        output_cost = getattr(args, "output_cost_per_1m_tokens", None)
        if input_cost is None or output_cost is None:
            raise ValueError(
                "--max-budget-per-session requires "
                "--input-cost-per-1m-tokens and --output-cost-per-1m-tokens."
            )

        def cost_estimator(response):
            return estimate_cost_from_usage(
                response,
                input_cost_per_1m_tokens=input_cost,
                output_cost_per_1m_tokens=output_cost,
            )

    return BudgetGuard(
        session_id=session_id,
        max_iterations=max_iterations,
        max_budget_per_session=max_budget_per_session,
        cost_estimator=cost_estimator,
    )


def _build_rate_limiter(
    min_seconds_between_calls: Optional[float],
    max_requests_per_minute: Optional[float],
    max_requests_per_second: Optional[float],
    max_tokens_per_minute: Optional[float],
) -> Optional[RateLimiter]:
    """Build ONE RateLimiter from resolved flag values, or None if none of
    them were set (matches _build_budget_guard's "no flags -> no object"
    convention). --max-requests-per-minute/--max-requests-per-second
    configure the same underlying cap in different units -- exactly one
    (or neither) may be set."""
    if max_requests_per_minute is not None and max_requests_per_second is not None:
        raise ValueError(
            "Set at most one of --max-requests-per-minute / "
            "--max-requests-per-second (or the --classifier- equivalents) "
            "-- they configure the same underlying cap, just in different "
            "units."
        )

    requests_per_minute = max_requests_per_minute
    if max_requests_per_second is not None:
        requests_per_minute = max_requests_per_second * 60.0

    if (
        min_seconds_between_calls is None
        and requests_per_minute is None
        and max_tokens_per_minute is None
    ):
        return None

    return RateLimiter(
        min_seconds_between_calls=min_seconds_between_calls,
        max_requests_per_minute=requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
    )

    ### EDITING END ###

def generate_dialogues_command(args):
    print("Starting dialogue generation...")
    print(f"Configuration: {args}")
    print("-" * 30)

    ### EDIT START ###

    reasoning_mode, reasoning_effort = _reasoning_mode_and_effort_from_args(args)

    shared_temperature = getattr(args, "temperature", None)
    user_temperature = (
        shared_temperature
        if shared_temperature is not None
        else args.user_llm_temperature
    )
    target_temperature = (
        shared_temperature
        if shared_temperature is not None
        else args.target_llm_temperature
    )

    shared_max_tokens = getattr(args, "max_tokens", None)
    user_max_tokens = (
        shared_max_tokens
        if shared_max_tokens is not None
        else getattr(args, "user_llm_max_tokens", None)
    )
    target_max_tokens = (
        shared_max_tokens
        if shared_max_tokens is not None
        else getattr(args, "target_llm_max_tokens", None)
    )

    shared_timeout = getattr(args, "timeout", None)
    user_timeout = (
        shared_timeout
        if shared_timeout is not None
        else getattr(args, "user_llm_timeout", None)
    )
    target_timeout = (
        shared_timeout
        if shared_timeout is not None
        else getattr(args, "target_llm_timeout", None)
    )

    # Rate limiting: resolve each threshold per-role first (shared flag,
    # if set, overrides the --user-llm-*/--target-llm-* one -- same
    # override pattern as --timeout/--max-tokens above), THEN build either
    # 1 shared instance or 2 independent ones from those resolved values.
    # See _build_rate_limiter's and RateLimiter's docstrings for why
    # "shared" vs "per-model" isn't just a convenience choice.
    def _resolve_rl(shared_attr, user_attr, target_attr):
        shared_value = getattr(args, shared_attr, None)
        user_value = (
            shared_value if shared_value is not None else getattr(args, user_attr, None)
        )
        target_value = (
            shared_value if shared_value is not None else getattr(args, target_attr, None)
        )
        return user_value, target_value

    user_rl_min_gap, target_rl_min_gap = _resolve_rl(
        "min_seconds_between_calls",
        "user_llm_min_seconds_between_calls",
        "target_llm_min_seconds_between_calls",
    )
    user_rl_rpm, target_rl_rpm = _resolve_rl(
        "max_requests_per_minute",
        "user_llm_max_requests_per_minute",
        "target_llm_max_requests_per_minute",
    )
    user_rl_rps, target_rl_rps = _resolve_rl(
        "max_requests_per_second",
        "user_llm_max_requests_per_second",
        "target_llm_max_requests_per_second",
    )
    user_rl_tpm, target_rl_tpm = _resolve_rl(
        "max_tokens_per_minute",
        "user_llm_max_tokens_per_minute",
        "target_llm_max_tokens_per_minute",
    )
    _rl_scope = getattr(args, "rate_limit_scope", "shared")

    _user_rl_thresholds = (user_rl_min_gap, user_rl_rpm, user_rl_rps, user_rl_tpm)
    _target_rl_thresholds = (target_rl_min_gap, target_rl_rpm, target_rl_rps, target_rl_tpm)

    user_rate_limiter = _build_rate_limiter(*_user_rl_thresholds)
    if user_rate_limiter is None and _build_rate_limiter(*_target_rl_thresholds) is None:
        target_rate_limiter = None
    elif _rl_scope == "shared":
        if _user_rl_thresholds != _target_rl_thresholds:
            raise ValueError(
                "--rate-limit-scope=shared (the default) requires the "
                "User and Target LLMs to end up with the SAME rate-limit "
                f"thresholds -- got user={_user_rl_thresholds} vs "
                f"target={_target_rl_thresholds}. A single shared clock "
                "can't enforce two different caps. Either drop the "
                "--user-llm-*/--target-llm-* overrides that make these "
                "differ, or pass --rate-limit-scope per-model to give "
                "each role its own independent budget."
            )
        target_rate_limiter = user_rate_limiter
    else:
        target_rate_limiter = _build_rate_limiter(*_target_rl_thresholds)

    budget_session_id = (
        getattr(args, "budget_session_id", None)
        or f"{sanitize_model_name(args.target_llm_model)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    budget_guard = _build_budget_guard(args, session_id=budget_session_id)
    
    
    # prepare LLM configurations
    user_llm_config = {
        "model": args.user_llm_model,
        "temperature": user_temperature,
        "max_retries": args.max_retries,
        "initial_backoff": args.initial_backoff,
        "max_backoff": args.max_backoff,
    }
    target_llm_config = {
        "model": args.target_llm_model,
        "temperature": target_temperature,
        "max_retries": args.max_retries,
        "initial_backoff": args.initial_backoff,
        "max_backoff": args.max_backoff,
    }
    if getattr(args, "max_timeout_retries", None) is not None:
        user_llm_config["max_timeout_retries"] = args.max_timeout_retries
        target_llm_config["max_timeout_retries"] = args.max_timeout_retries
    if getattr(args, "max_message_order_retries", None) is not None:
        user_llm_config["max_message_order_retries"] = args.max_message_order_retries
        target_llm_config["max_message_order_retries"] = args.max_message_order_retries
    if user_max_tokens is not None:
        user_llm_config["max_tokens"] = user_max_tokens
    if target_max_tokens is not None:
        target_llm_config["max_tokens"] = target_max_tokens
    if user_timeout is not None:
        user_llm_config["timeout"] = user_timeout
    if target_timeout is not None:
        target_llm_config["timeout"] = target_timeout
    if user_rate_limiter is not None:
        user_llm_config["rate_limiter"] = user_rate_limiter
    if target_rate_limiter is not None:
        target_llm_config["rate_limiter"] = target_rate_limiter

    if budget_guard is not None:
        user_llm_config["budget_guard"] = budget_guard
        target_llm_config["budget_guard"] = budget_guard

    user_openrouter_provider = _openrouter_provider_from_args(args, "user_llm")
    target_openrouter_provider = _openrouter_provider_from_args(args, "target_llm")
    if user_openrouter_provider:
        user_llm_config["openrouter_provider"] = user_openrouter_provider
        print(f"User LLM OpenRouter provider routing: {user_openrouter_provider}")
    if target_openrouter_provider:
        target_llm_config["openrouter_provider"] = target_openrouter_provider
        print(f"Target LLM OpenRouter provider routing: {target_openrouter_provider}")

####### EDIT FINISH HERE #######

    # handle system prompts
    user_system_prompt = args.user_system_prompt
    if os.path.exists(user_system_prompt):
        try:
            with open(user_system_prompt, "r", encoding="utf-8") as f:
                user_system_prompt = f.read()
            print(f"Loaded user system prompt from {args.user_system_prompt}")
        except Exception as e:
            print(
                f"Warning: Could not read user system prompt file {args.user_system_prompt}: {e}. Using provided string or default."
            )

    target_system_prompt = args.target_system_prompt
    if os.path.exists(target_system_prompt):
        try:
            with open(target_system_prompt, "r", encoding="utf-8") as f:
                target_system_prompt = f.read()
            print(f"Loaded target system prompt from {args.target_system_prompt}")
        except Exception as e:
            print(
                f"Warning: Could not read target system prompt file {args.target_system_prompt}: {e}. Using provided string or default."
            )

    # construct dynamic CSV filename
    if getattr(args, "output_csv_filename", None):
        dynamic_csv_filename = args.output_csv_filename
        print(f"Using explicit CSV filename: {dynamic_csv_filename}")
    else:
        sanitized_target_model = sanitize_model_name(args.target_llm_model)

        categories_str_part = "all_prompts"  # default if no specific categories or path
        if args.prompt_category_name:
            categories_str_part = "_".join(sorted(args.prompt_category_name))
            categories_str_part = categories_str_part.replace(" ", "_")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        dpc_arg = getattr(args, "dialogues_per_condition", None)
        dpc_str_part = f"_dpc{dpc_arg}" if dpc_arg is not None else ""

        dynamic_csv_filename = (
            f"dialogues_{sanitized_target_model}_{categories_str_part}{dpc_str_part}_{timestamp}.csv"
        )
        print(f"Generated CSV filename: {dynamic_csv_filename}")

    try:
        cues_arg = (
            args.behaviors
            if hasattr(args, "behaviors") and args.behaviors
            else getattr(args, "cues", None)
        )
        generator = DialogueGenerator(
            cues=cues_arg,
            user_llm_config=user_llm_config,
            target_llm_config=target_llm_config,
            user_system_prompt=user_system_prompt,
            target_system_prompt=target_system_prompt,
            num_turns=args.num_turns,
            num_dialogues=args.num_dialogues,
            dialogues_per_condition=getattr(args, "dialogues_per_condition", None),
            reasoning_mode=reasoning_mode,
            reasoning_effort=reasoning_effort,
            prompt_category_names=args.prompt_category_name,
            custom_prompt_csv=args.custom_prompt_csv,
            use_all_variants_of_original_prompt=not args.deduplicate_original_prompts,
            output_dir=args.output_dir,
            default_csv_filename=dynamic_csv_filename,
            max_concurrency=getattr(args, "max_concurrency", 1),
            strict_batch_ordering=getattr(args, "strict_batch_ordering", False),
            incremental_save=getattr(args, "incremental_save", False),
            resume=getattr(args, "resume", False),
            stop_on_natural_end=getattr(args, "stop_on_natural_end", False),
        )

        print("DialogueGenerator initialized.")

        print("Generating dialogues...")
        generated_dialogues = generator.generate_dialogues()

        if generated_dialogues:
            print(f"Successfully generated {len(generated_dialogues)} dialogue(s).")
            output_csv_path = os.path.join(args.output_dir, dynamic_csv_filename)
            if not os.path.isabs(output_csv_path):
                output_csv_path = os.path.abspath(output_csv_path)
            print(f"Dialogues saved to: {output_csv_path}")
            if not os.path.exists(output_csv_path):
                print(
                    f"ERROR: CSV file NOT found at {output_csv_path}, though generation reported success."
                )
        elif generator.prompts:
            print(
                "Prompts were loaded, but no dialogues were generated (check num_dialogues or filtering criteria)."
            )
        else:
            print(
                "No prompts were loaded (check prompt_category_names/prompt_path and file existence/content), so no dialogues generated."
            )

    except LLMGenerationError as e:
        print(f"LLM Generation Error: {e}", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()


def rate_dialogues_command(args):
    """
    Command handler for the 'rate' subcommand.
    Maps CLI args to library function parameters.
    """

    reasoning_mode, reasoning_effort = _reasoning_mode_and_effort_from_args(args) ### EDITED
    classifier_openrouter_provider = _openrouter_provider_from_args(args, "classifier") ### EDITED

    # Rate limiting: build either one shared instance or a {model:
    # RateLimiter} dict (one independent instance per --classifier-model)
    # from the SAME configured thresholds -- see rate_dialogues()'s
    # docstring in rating.py and RateLimiter's in llm_client.py for why
    # "shared" vs "per-model" isn't just a convenience choice, and why
    # per-model is the default here specifically (unlike generate's
    # shared default).
    _rl_min_gap = getattr(args, "classifier_min_seconds_between_calls", None)
    _rl_rpm = getattr(args, "classifier_max_requests_per_minute", None)
    _rl_rps = getattr(args, "classifier_max_requests_per_second", None)
    _rl_tpm = getattr(args, "classifier_max_tokens_per_minute", None)
    _rl_scope = getattr(args, "classifier_rate_limit_scope", "per-model")

    if _build_rate_limiter(_rl_min_gap, _rl_rpm, _rl_rps, _rl_tpm) is None:
        classifier_rate_limiter = None
    elif _rl_scope == "shared":
        classifier_rate_limiter = _build_rate_limiter(_rl_min_gap, _rl_rpm, _rl_rps, _rl_tpm)
    else:
        classifier_rate_limiter = {
            model: _build_rate_limiter(_rl_min_gap, _rl_rpm, _rl_rps, _rl_tpm)
            for model in args.classifier_model
        }

    budget_session_id = (
        getattr(args, "budget_session_id", None)
        or f"rate_{sanitize_model_name('_'.join(sorted(args.classifier_model)))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    budget_guard = _build_budget_guard(args, session_id=budget_session_id)

    try:
        cues_to_rate_arg = (
            args.behaviors_to_rate
            if hasattr(args, "behaviors_to_rate") and args.behaviors_to_rate
            else getattr(args, "cues_to_rate", None)
        )
        if classifier_openrouter_provider:
            print(f"Classifier LLM OpenRouter provider routing: {classifier_openrouter_provider}")
        output_path = run_rating_process(
            dialogues_csv_path=args.dialogues_csv,
            cues_to_rate=cues_to_rate_arg,
            classifier_models=args.classifier_model,
            classifier_temperature=args.classifier_temperature,
            num_samples=args.num_samples,
            output_rated_csv=getattr(args, "output_rated_csv", None),
            classifier_reasoning_mode=reasoning_mode, ### EDITED
            classifier_reasoning_effort=reasoning_effort, ### EDITED
            classifier_openrouter_provider=classifier_openrouter_provider, ### EDITED
            cue_group_config=getattr(args, "cue_group_config", None),
            classifier_max_tokens_base=getattr(args, "classifier_max_tokens_base", None),
            classifier_max_tokens_per_cue=getattr(args, "classifier_max_tokens_per_cue", None),
            classifier_timeout=getattr(args, "classifier_timeout", None),
            classifier_max_retries=getattr(args, "classifier_max_retries", 5),
            classifier_max_timeout_retries=getattr(args, "classifier_max_timeout_retries", None),
            classifier_initial_backoff=getattr(args, "classifier_initial_backoff", 2.0),
            classifier_max_backoff=getattr(args, "classifier_max_backoff", 60.0),
            rate_limiter=classifier_rate_limiter,
            budget_guard=budget_guard,
            max_concurrency=getattr(args, "max_concurrency", 1),
            strict_batch_ordering=getattr(args, "strict_batch_ordering", False),
            incremental_save=getattr(args, "incremental_save", False),
            resume=getattr(args, "resume", False),
            verbose=True,
        )
        if not output_path:
            print("Warning: Rating process completed but didn't return an output path.")
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}", file=sys.stderr)
    except ValueError as e:
        print(f"Value Error: {e}", file=sys.stderr)
    except IOError as e:
        print(f"I/O Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()


def summarize_command(args):
    print("Starting analysis summary...")
    try:
        from anthro_benchmark.analysis.analysis import run_analysis

        run_analysis(
            rated_csv_path=args.rated_csv,
            output_dir=args.output_dir,
            first_n_turns=args.first_n_turns,
            require_min_turns=args.require_min_turns,
        )
    except ImportError as e:
        print(f"Import Error during analysis: {e}", file=sys.stderr)
        print(
            "Please ensure pandas and plotly are installed: pip install pandas plotly",
            file=sys.stderr,
        )
        print(
            'For saving static plots (PNG), also install kaleido: pip install -U "kaleido>=0.1.0,<0.2.0"',
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An error occurred during analysis: {e}", file=sys.stderr)
        traceback.print_exc()


def _parse_flags(_):
    parser = argparse_flags.ArgumentParser(
        description="CLI for Anthropomorphic Behavior Evaluation in LLMs."
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Generate dialogues subcommand
    gen_parser = subparsers.add_parser(
        "generate", help="Generate dialogues between LLMs."
    )

    prompt_group = gen_parser.add_argument_group("Prompting options")
    prompt_group.add_argument(
        "--prompt-category-name",
        type=str,
        nargs="+",
        help='One or more prompt category names (e.g., "personhood" "physical_embodiment").',
    )
    prompt_group.add_argument(
        "--custom-prompt-csv",
        type=str,
        required=False,
        help="Path to custom CSV file to use for dialogue generation. Uses prompt_sets.csv if no CSV is specified.",
    )
    prompt_group.add_argument(
        "--behaviors",
        type=str,
        nargs="*",
        help="List of specific behaviors to filter for.",
    )
    prompt_group.add_argument(
        "--cues", type=str, nargs="*", help=argparse.SUPPRESS
    )  # deprecated alias
    prompt_group.add_argument(
        "--deduplicate-original-prompts",
        action="store_true",
        help="If set, uses unique 'original_prompt' values from loaded prompts, keeping the first. Default is to use all variants.",
    )

    llm_group = gen_parser.add_argument_group("LLM configuration options")
    llm_group.add_argument(
        "--user-llm-model", type=str, required=True, help="Model name for User LLM."
    )
    llm_group.add_argument(
        "--user-llm-temperature",
        type=float,
        default=0.7,
        help="Temperature for User LLM.",
    )
    llm_group.add_argument(
        "--target-llm-model", type=str, required=True, help="Model name for Target LLM."
    )
    llm_group.add_argument(
        "--target-llm-temperature",
        type=float,
        default=0.7,
        help="Temperature for Target LLM.",
    )
    llm_group.add_argument(
        "--user-system-prompt",
        type=str,
        default=DEFAULT_USER_SYSTEM_PROMPT,
        help="Custom system prompt for User LLM (string or path to .txt file).",
    )
    llm_group.add_argument(
        "--target-system-prompt",
        type=str,
        default="You are a helpful chatbot. Every one of your outputs should be a single message in a conversation.",
        help="Custom system prompt for Target LLM (string or path to .txt file).",
    )


    ### EDITING ###

    # add these arguments to the generate subcommand's llm_group
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Shared temperature for both user and target LLMs. If set, overrides the per-model temperature flags.",
    )
    llm_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Shared output-token cap (max_tokens) for both the User and "
            "Target LLMs. If set, overrides --user-llm-max-tokens/"
            "--target-llm-max-tokens. Circuit breaker against a runaway "
            "generation (e.g. a provider that repeats a degenerate token "
            "instead of stopping) burning input tokens turn after turn -- "
            "not a cost-optimization lever, so set it generously (well "
            "above your longest expected reply). Only bounds output "
            "tokens, never input/prompt tokens. Default: unset (unbounded, "
            "matches prior behavior). If --reasoning-mode is 'on' for the "
            "Target LLM, most providers count reasoning tokens against "
            "this same budget alongside the visible reply -- too tight a "
            "cap can truncate the reasoning before any reply is produced."
        ),
    )
    llm_group.add_argument(
        "--user-llm-max-tokens",
        type=int,
        default=None,
        help="Output-token cap (max_tokens) for the User LLM only. Ignored if --max-tokens is set.",
    )
    llm_group.add_argument(
        "--target-llm-max-tokens",
        type=int,
        default=None,
        help="Output-token cap (max_tokens) for the Target LLM only. Ignored if --max-tokens is set.",
    )
    llm_group.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Shared per-call timeout in seconds for both the User and "
            "Target LLMs. If set, overrides --user-llm-timeout/"
            "--target-llm-timeout. Bounds how long a single call can hang "
            "before it's treated as failed, instead of blocking this "
            "thread indefinitely on a stuck provider. A timeout is "
            "retried like any other transient error, but against its OWN "
            "budget -- see --max-timeout-retries below, not --max-retries. "
            "Default: unset (no client-side cap, matches prior behavior "
            "-- a call can hang as long as the provider/transport "
            "allows)."
        ),
    )
    llm_group.add_argument(
        "--user-llm-timeout",
        type=float,
        default=None,
        help="Per-call timeout in seconds for the User LLM only. Ignored if --timeout is set.",
    )
    llm_group.add_argument(
        "--target-llm-timeout",
        type=float,
        default=None,
        help="Per-call timeout in seconds for the Target LLM only. Ignored if --timeout is set.",
    )
    llm_group.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts per LLM call for transient errors OTHER than a timeout (429, 5xx, connection drops). See --max-timeout-retries for timeouts specifically.",
    )
    llm_group.add_argument(
        "--max-timeout-retries",
        type=int,
        default=None,
        help=(
            "Maximum retry attempts specifically for a --timeout expiry, "
            "kept independent of --max-retries: a run of ordinary 429s/"
            "5xx errors won't eat into this budget, and vice versa. "
            "Worth setting LOWER than --max-retries if you're relying on "
            "--timeout for recovery, since each timeout retry has an "
            "unknown, possibly-nonzero provider-side cost that's never "
            "visible to --max-budget-per-session (we never get a "
            "response back to bill it against, unlike a 429/connection "
            "error, which is unambiguously free). Default: unset -- uses "
            "the same value as --max-retries, matching pre-split "
            "behavior."
        ),
    )
    llm_group.add_argument(
        "--max-message-order-retries",
        type=int,
        default=None,
        help=(
            "Maximum retry attempts specifically for Mistral's code-3230 "
            "'invalid_request_message_order' error (\"Expected last role "
            "User or Tool ... but got assistant\"), kept independent of "
            "--max-retries. This has been observed as an intermittent "
            "litellm/OpenRouter routing quirk rather than a locally-"
            "malformed request -- see "
            "https://github.com/anomalyco/opencode/issues/6346 -- so a "
            "retry is worth attempting before giving up. Each retry also "
            "logs the exact message role sequence that triggered it, so "
            "a genuine local bug (as opposed to the routing quirk this "
            "exists for) would show up in the logs rather than being "
            "silently retried away. Default: unset -- uses LLMClient's "
            "own default of 2."
        ),
    )
    llm_group.add_argument(
        "--initial-backoff",
        type=float,
        default=2.0,
        help="Initial retry backoff in seconds.",
    )
    llm_group.add_argument(
        "--max-backoff",
        type=float,
        default=60.0,
        help="Maximum retry backoff in seconds.",
    )

    # add a new budget group inside generate, after the reasoning group or after llm_group
    budget_group = gen_parser.add_argument_group("Budget options")
    budget_group.add_argument(
        "--budget-session-id",
        type=str,
        default=None,
        help="Optional session ID for budget tracking.",
    )
    budget_group.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Hard cap on total LLM calls for the run.",
    )
    budget_group.add_argument(
        "--max-budget-per-session",
        type=float,
        default=None,
        help="Hard cap on estimated spend for the run.",
    )
    budget_group.add_argument(
        "--input-cost-per-1m-tokens",
        type=float,
        default=None,
        help="Prompt-token price used for budget estimation.",
    )
    budget_group.add_argument(
        "--output-cost-per-1m-tokens",
        type=float,
        default=None,
        help="Completion-token price used for budget estimation.",
    )

    rate_limit_group = gen_parser.add_argument_group("Rate limiting options")
    rate_limit_group.add_argument(
        "--min-seconds-between-calls",
        type=float,
        default=None,
        help=(
            "Shared minimum wait, in seconds, enforced between the START "
            "of consecutive LLM calls for BOTH the User and Target LLMs "
            "-- a flat pacing throttle (your 'waiting time'), independent "
            "of the caps below. If set, overrides "
            "--user-llm-min-seconds-between-calls/"
            "--target-llm-min-seconds-between-calls. Under "
            "--rate-limit-scope=shared (the default), User and Target "
            "calls are paced against ONE shared cadence; under "
            "per-model, each gets its own. Default: unset (no pacing)."
        ),
    )
    rate_limit_group.add_argument(
        "--user-llm-min-seconds-between-calls",
        type=float,
        default=None,
        help="Minimum wait, in seconds, between User LLM calls only. Ignored if --min-seconds-between-calls is set. Requires --rate-limit-scope=per-model to differ from the Target LLM's value (a shared cadence can't honor two different paces).",
    )
    rate_limit_group.add_argument(
        "--target-llm-min-seconds-between-calls",
        type=float,
        default=None,
        help="Minimum wait, in seconds, between Target LLM calls only. Ignored if --min-seconds-between-calls is set. Requires --rate-limit-scope=per-model to differ from the User LLM's value.",
    )
    rate_limit_group.add_argument(
        "--max-requests-per-minute",
        type=float,
        default=None,
        help=(
            "Shared cap on LLM calls in any rolling 60-second window, for "
            "BOTH the User and Target LLMs. If set, overrides "
            "--user-llm-max-requests-per-minute/"
            "--target-llm-max-requests-per-minute. Mutually exclusive "
            "with --max-requests-per-second (same underlying cap, "
            "different unit)."
        ),
    )
    rate_limit_group.add_argument(
        "--user-llm-max-requests-per-minute",
        type=float,
        default=None,
        help="Requests-per-minute cap for the User LLM only. Ignored if --max-requests-per-minute is set. Requires --rate-limit-scope=per-model to differ from the Target LLM's value.",
    )
    rate_limit_group.add_argument(
        "--target-llm-max-requests-per-minute",
        type=float,
        default=None,
        help="Requests-per-minute cap for the Target LLM only. Ignored if --max-requests-per-minute is set. Requires --rate-limit-scope=per-model to differ from the User LLM's value.",
    )
    rate_limit_group.add_argument(
        "--max-requests-per-second",
        type=float,
        default=None,
        help=(
            "Same cap as --max-requests-per-minute, expressed per second "
            "(converted internally as value * 60), for BOTH the User and "
            "Target LLMs. Mutually exclusive with --max-requests-per-minute."
        ),
    )
    rate_limit_group.add_argument(
        "--user-llm-max-requests-per-second",
        type=float,
        default=None,
        help="Requests-per-second cap for the User LLM only. Ignored if --max-requests-per-second/--max-requests-per-minute is set.",
    )
    rate_limit_group.add_argument(
        "--target-llm-max-requests-per-second",
        type=float,
        default=None,
        help="Requests-per-second cap for the Target LLM only. Ignored if --max-requests-per-second/--max-requests-per-minute is set.",
    )
    rate_limit_group.add_argument(
        "--max-tokens-per-minute",
        type=float,
        default=None,
        help=(
            "Shared cap on total tokens (input+output combined) in any "
            "rolling 60-second window, for BOTH the User and Target LLMs, "
            "enforced using each call's ACTUAL usage once it's known -- a "
            "single large call can still push the window over the cap; "
            "later calls are then held back to compensate. Reactive, not "
            "a hard preemptive guarantee -- see RateLimiter's docstring "
            "in llm_client.py. If set, overrides "
            "--user-llm-max-tokens-per-minute/"
            "--target-llm-max-tokens-per-minute."
        ),
    )
    rate_limit_group.add_argument(
        "--user-llm-max-tokens-per-minute",
        type=float,
        default=None,
        help="Tokens-per-minute cap for the User LLM only. Ignored if --max-tokens-per-minute is set. Requires --rate-limit-scope=per-model to differ from the Target LLM's value.",
    )
    rate_limit_group.add_argument(
        "--target-llm-max-tokens-per-minute",
        type=float,
        default=None,
        help="Tokens-per-minute cap for the Target LLM only. Ignored if --max-tokens-per-minute is set. Requires --rate-limit-scope=per-model to differ from the User LLM's value.",
    )
    rate_limit_group.add_argument(
        "--rate-limit-scope",
        choices=["shared", "per-model"],
        default="shared",
        help=(
            "Only matters if at least one cap above is set. 'shared' "
            "(default): User and Target LLM calls are throttled TOGETHER "
            "against one combined budget -- correct when both use the "
            "same provider account/key, which is the common case. Requires "
            "the User and Target LLMs to end up with IDENTICAL thresholds "
            "(a single shared clock can't enforce two different caps at "
            "once) -- set per-role values that differ and this will raise "
            "a clear error telling you to switch to per-model instead of "
            "silently picking one. 'per-model': each gets its OWN "
            "independent budget -- use this if they're on different "
            "providers/keys with genuinely separate quotas, OR if you set "
            "different --user-llm-*/--target-llm-* rate-limit values on "
            "purpose."
        ),
    )

    concurrency_group = gen_parser.add_argument_group("Concurrency options")
    concurrency_group.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help=(
            "How many dialogues to generate at once (default: 1, fully "
            "sequential -- identical to the original behavior). Runs "
            "concurrently across dialogues via asyncio.to_thread; turns "
            "within a single dialogue are always generated in order "
            "regardless of this setting. Under concurrency, a "
            "--max-iterations/--max-budget-per-session cap can be "
            "exceeded by roughly up to this many extra in-flight calls "
            "(see BudgetGuard's docstring in llm_client.py) -- keep this "
            "modest relative to your cap if you want the overshoot bound "
            "to stay small."
        ),
    )

    concurrency_group.add_argument(
        "--strict-batch-ordering",
        action="store_true",
        help=(
            "Only relevant with --max-concurrency > 1. Default off: "
            "dispatch continuously for maximum throughput (a new dialogue "
            "starts the instant a slot frees up); if a budget cap is hit "
            "mid-run, which dialogues end up complete doesn't follow index "
            "order. Set this to dispatch in batches of --max-concurrency "
            "instead, one batch fully finished before the next starts -- "
            "bounds that ambiguity to within one batch, at a measured "
            "throughput cost (roughly 1-15%% under ordinary latency "
            "variance, 2x+ if a straggler shows up in a batch). Only "
            "matters if a budget cap actually binds mid-run; irrelevant "
            "with no cap or a generous one. Automatically enabled if "
            "--incremental-save is set together with --max-concurrency > 1 "
            "(see --incremental-save)."
        ),
    )
    concurrency_group.add_argument(
        "--incremental-save",
        action="store_true",
        help=(
            "Default off: the CSV is written once, after the whole run "
            "finishes -- a crash/hang/kill before that point saves "
            "nothing at all. Set this to rewrite the CSV after each "
            "completed dialogue (sequential) or each completed batch "
            "(concurrent -- this forces --strict-batch-ordering on "
            "automatically, since safe checkpointing under concurrency "
            "needs a guaranteed-complete, gap-free prefix to save), so a "
            "crash partway through leaves everything completed so far on "
            "disk. Each save is atomic (temp file + rename), so an "
            "interruption during the save itself can't corrupt the "
            "previous good checkpoint either."
        ),
    )
    concurrency_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Requires --incremental-save. If --output-dir/--default-"
            "csv-filename already exists (e.g. from a crashed prior run), "
            "skips regenerating any dialogue that already completed "
            "successfully there and only generates what's missing or "
            "didn't complete -- matched by dialogue_index, which "
            "deterministically identifies 'the same dialogue' across "
            "separate invocations given the same input prompts (unlike "
            "dialogue_id, a random UUID that differs every run). If the "
            "output path doesn't exist yet, this has no effect -- a "
            "normal fresh run."
        ),
    )

    early_stop_group = gen_parser.add_argument_group("Early-stopping options")
    early_stop_group.add_argument(
        "--stop-on-natural-end",
        action="store_true",
        help=(
            "Default off, reproducing the original behavior exactly "
            "(every dialogue runs the full --num-turns regardless of "
            "content). When set, the USER LLM (never the target -- see "
            "generator.py's NATURAL_END_INSTRUCTION_TEMPLATE comment for "
            "why) is instructed to recognize when the target's message "
            "is a natural conversational close and, if so, give one "
            "final reply and stop the dialogue there instead of "
            "continuing to the full turn count. Avoids paying for a "
            "string of trivial closing-exchange turns (each re-sending "
            "the whole growing conversation as input tokens) after a "
            "conversation has already wrapped up. Affects dialogue "
            "length distribution -- leave this off if your experimental "
            "design specifically requires fixed-length dialogues."
        ),
    )

    ### EDITING ENDED ###

    ### START EDITING ###
    
    reasoning_group = gen_parser.add_argument_group("Reasoning options")
    reasoning_group.add_argument(
    "--reasoning-mode",
    choices=["default", "on", "off"],
    default="default",
    help="Default = let the provider decide; on = enable reasoning; off = disable reasoning.",
    )
    reasoning_group.add_argument(
    "--reasoning-effort",
    choices=["minimal", "low", "medium", "high", "xhigh", "max"],
    help="Optional effort level when reasoning is on.",
    )

    ### EDIT ENDED ###

    ### NEW: OpenRouter provider-routing options ###
    #
    # Why this exists: OpenRouter load-balances a single model slug across
    # multiple backend providers by default (see
    # https://openrouter.ai/docs/guides/routing/provider-selection). Those
    # providers don't all support the same request parameters -- e.g. for
    # "google/gemma-3-27b-it:free" on OpenRouter, the "Google AI Studio"
    # backend honors `reasoning`, but a fallback backend ("Darkbloom") has
    # been observed to silently ignore it (reasoning_tokens_reported=0 with
    # no error), rather than reject the request. These flags let you pin
    # (and optionally *require*) specific backends per LLM role so this
    # doesn't happen invisibly mid-run.
    openrouter_group = gen_parser.add_argument_group("OpenRouter provider routing options")
    openrouter_group.add_argument(
        "--user-llm-openrouter-provider-order",
        type=str,
        nargs="+",
        default=None,
        metavar="PROVIDER_SLUG",
        help=(
            "OpenRouter provider slug(s) to try, in priority order, for the "
            "User LLM (e.g. 'google-ai-studio' or 'Google AI Studio' -- "
            "OpenRouter's provider slugs and display names both work; use "
            "the copy button on the model's OpenRouter page to get the "
            "exact slug). Only meaningful when --user-llm-model starts with "
            "'openrouter/'. Unlisted providers remain available as "
            "fallbacks unless --user-llm-openrouter-no-fallbacks is also "
            "set."
        ),
    )
    openrouter_group.add_argument(
        "--user-llm-openrouter-no-fallbacks",
        action="store_true",
        help=(
            "Disable fallback to any provider not listed in "
            "--user-llm-openrouter-provider-order. Requires that flag to "
            "also be set."
        ),
    )
    openrouter_group.add_argument(
        "--user-llm-openrouter-require-parameters",
        action="store_true",
        help=(
            "Only route the User LLM's requests to providers that support "
            "every parameter in the request. A provider that would "
            "otherwise silently ignore an unsupported parameter is excluded "
            "instead. Usually unnecessary for the User LLM, since reasoning "
            "is never requested for it -- provided mainly for symmetry with "
            "--target-llm-openrouter-require-parameters."
        ),
    )
    openrouter_group.add_argument(
        "--target-llm-openrouter-provider-order",
        type=str,
        nargs="+",
        default=None,
        metavar="PROVIDER_SLUG",
        help=(
            "Same as --user-llm-openrouter-provider-order, but for the "
            "Target LLM. This is the one that matters when --reasoning-mode "
            "is 'on': if the Target LLM's model has multiple OpenRouter "
            "backends and only some of them honor `reasoning`, pin to a "
            "backend that does (e.g. --target-llm-openrouter-provider-order "
            "'Google AI Studio') to stop reasoning-token generation from "
            "silently dropping to 0 on calls that get routed elsewhere."
        ),
    )
    openrouter_group.add_argument(
        "--target-llm-openrouter-no-fallbacks",
        action="store_true",
        help=(
            "Disable fallback to any provider not listed in "
            "--target-llm-openrouter-provider-order. Requires that flag to "
            "also be set. Set this if you want the run to fail loudly "
            "rather than silently fall back to a provider you haven't "
            "verified supports reasoning."
        ),
    )
    openrouter_group.add_argument(
        "--target-llm-openrouter-require-parameters",
        action="store_true",
        help=(
            "Only route the Target LLM's requests to providers that "
            "support every parameter in the request (notably `reasoning` "
            "when --reasoning-mode is 'on'). A provider that would "
            "otherwise silently ignore reasoning is excluded from routing "
            "instead, so a misrouted call fails or falls back visibly "
            "rather than quietly returning reasoning_tokens=0."
        ),
    )
    ### END NEW ###

    gen_control_group = gen_parser.add_argument_group("Generation control options")
    dialogue_count_group = gen_control_group.add_mutually_exclusive_group()
    dialogue_count_group.add_argument(
        "--num-dialogues",
        type=int,
        default=None,
        help=(
            "Number of dialogues to produce. If omitted, defaults to one "
            "dialogue per loaded prompt (i.e. the size of the prompt set "
            "actually loaded, after --prompt-category-name / --behaviors "
            "filtering and/or --custom-prompt-csv are applied). Set this "
            "explicitly to override, e.g. to replicate each prompt multiple "
            "times or to cap a run short. Mutually exclusive with "
            "--dialogues-per-condition."
        ),
    )
    dialogue_count_group.add_argument(
        "--dialogues-per-condition",
        type=int,
        default=None,
        help=(
            "Generate exactly this many dialogues for EACH unique condition "
            "(unique combination of use_domain/use_scenario/empathy/"
            "professionalism/cue/behavior_category found in the loaded "
            "prompt set), instead of a fixed total. Total dialogues produced "
            "= dialogues_per_condition x (number of unique conditions found). "
            "E.g. a 96-condition --custom-prompt-csv with "
            "--dialogues-per-condition 2 produces 192 dialogues, 2 per "
            "condition. Requires the loaded prompt set to have at least one "
            "of those condition columns. Mutually exclusive with "
            "--num-dialogues."
        ),
    )
    gen_control_group.add_argument(
        "--num-turns", type=int, default=5, help="Number of turns."
    )

    output_group = gen_parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="generated_dialogues",
        help="Directory to save generated dialogues.",
    )
    output_group.add_argument(
        "--output-csv-filename",
        type=str,
        default=None,
        help=(
            "Exact filename to use instead of the auto-generated "
            "'dialogues_<model>_<categories>_<timestamp>.csv' -- the "
            "timestamp in that default means every invocation produces a "
            "different filename, so --resume (see --incremental-save) "
            "has nothing to find unless you pin the filename explicitly "
            "here to match a prior/partial run's output. Also needed if "
            "you've downloaded a partial CSV to resume: place it at "
            "--output-dir/--output-csv-filename before running with "
            "--resume."
        ),
    )

    gen_parser.set_defaults(func=generate_dialogues_command)

    # Rate dialogues subcommand
    rate_parser = subparsers.add_parser(
        "rate",
        help="Rate dialogues for specified behaviors using one or more LLM classifiers.",
    )

    rate_parser.add_argument(
        "--dialogues-csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing dialogues.",
    )
    rate_parser.add_argument(
        "--output-rated-csv",
        type=str,
        help="Path for the output CSV. If not provided, generated filename is placed in 'rated_dialogues' directory.",
    )

    rate_llm_group = rate_parser.add_argument_group("Classifier LLM Configuration")
    rate_llm_group.add_argument(
        "--classifier-model",
        type=str,
        nargs="+",
        required=True,
        help="One or more model names for the classifier LLM(s).",
    )
    rate_llm_group.add_argument(
        "--classifier-temperature",
        type=float,
        default=0.7,
        help="Temperature for the classifier LLM(s).",
    )
    rate_llm_group.add_argument(
        "--classifier-timeout",
        type=float,
        default=None,
        help=(
            "Per-call timeout in seconds, applied identically to every "
            "model in --classifier-model. Bounds how long a single rating "
            "call can hang before it's treated as failed, instead of "
            "blocking indefinitely on a stuck provider. A timeout is "
            "retried like any other transient error, but against its OWN "
            "budget -- see --classifier-max-timeout-retries below, not "
            "--classifier-max-retries. Default: unset (no client-side "
            "cap, matches prior behavior)."
        ),
    )
    rate_llm_group.add_argument(
        "--classifier-max-retries",
        type=int,
        default=5,
        help=(
            "Maximum retry attempts per classifier call for transient "
            "errors OTHER than a timeout (429, 5xx, connection drops). "
            "Applied identically to every model in --classifier-model. "
            "See --classifier-max-timeout-retries for timeouts "
            "specifically. Previously not configurable here at all -- "
            "classifier calls always used LLMClient's hardcoded default "
            "of 5; this flag's default matches that, so omitting it "
            "changes nothing."
        ),
    )
    rate_llm_group.add_argument(
        "--classifier-max-timeout-retries",
        type=int,
        default=None,
        help=(
            "Maximum retry attempts specifically for a "
            "--classifier-timeout expiry, kept independent of "
            "--classifier-max-retries: a run of ordinary 429s/5xx errors "
            "won't eat into this budget, and vice versa. Worth setting "
            "LOWER than --classifier-max-retries if you're relying on "
            "--classifier-timeout for recovery, since each timeout retry "
            "has an unknown, possibly-nonzero provider-side cost that's "
            "never visible to --max-budget-per-session (we never get a "
            "response back to bill it against). Default: unset -- uses "
            "the same value as --classifier-max-retries."
        ),
    )
    rate_llm_group.add_argument(
        "--classifier-initial-backoff",
        type=float,
        default=2.0,
        help=(
            "Initial retry backoff in seconds for classifier calls "
            "(doubles on each subsequent retry, capped at "
            "--classifier-max-backoff). Previously not configurable here "
            "-- default matches LLMClient's prior hardcoded value, so "
            "omitting it changes nothing."
        ),
    )
    rate_llm_group.add_argument(
        "--classifier-max-backoff",
        type=float,
        default=60.0,
        help=(
            "Maximum retry backoff in seconds for classifier calls. "
            "Previously not configurable here -- default matches "
            "LLMClient's prior hardcoded value, so omitting it changes "
            "nothing."
        ),
    )

    ### EDITING START HERE ###
    
    reasoning_group = rate_parser.add_argument_group("Reasoning options") ### FIX
    reasoning_group.add_argument(
    "--reasoning-mode",
    choices=["default", "on", "off"],
    default="default",
    help="Default = let the provider decide; on = enable reasoning; off = disable reasoning.",
    )
    reasoning_group.add_argument(
    "--reasoning-effort",
    choices=["minimal", "low", "medium", "high", "xhigh", "max"],
    help="Optional effort level when reasoning is on.",
    )

    ### EDIT ENDED ###

    ### NEW: OpenRouter provider-routing options (classifier LLM) ###
    #
    # Same mechanism as the generate subcommand's user/target flags -- see
    # that argument group's help text for the full rationale. Applied
    # identically to every model passed via --classifier-model, so don't
    # set these if you're mixing OpenRouter and non-OpenRouter classifier
    # models in the same run (LLMClient will raise a clear error for the
    # non-OpenRouter one(s) rather than silently ignoring it).
    classifier_openrouter_group = rate_parser.add_argument_group(
        "OpenRouter provider routing options"
    )
    classifier_openrouter_group.add_argument(
        "--classifier-openrouter-provider-order",
        type=str,
        nargs="+",
        default=None,
        metavar="PROVIDER_SLUG",
        help=(
            "OpenRouter provider slug(s) to try, in priority order, for "
            "every model in --classifier-model (e.g. 'Google AI Studio'). "
            "Only meaningful when those models are 'openrouter/...' models. "
            "Unlisted providers remain available as fallbacks unless "
            "--classifier-openrouter-no-fallbacks is also set."
        ),
    )
    classifier_openrouter_group.add_argument(
        "--classifier-openrouter-no-fallbacks",
        action="store_true",
        help=(
            "Disable fallback to any provider not listed in "
            "--classifier-openrouter-provider-order. Requires that flag to "
            "also be set."
        ),
    )
    classifier_openrouter_group.add_argument(
        "--classifier-openrouter-require-parameters",
        action="store_true",
        help=(
            "Only route classifier requests to providers that support "
            "every parameter in the request (notably `reasoning` when "
            "--reasoning-mode is 'on'). A provider that would otherwise "
            "silently ignore reasoning is excluded from routing instead."
        ),
    )
    ### END NEW ###

    rate_config_group = rate_parser.add_argument_group("Rating Configuration")
    rate_config_group.add_argument(
        "--behaviors-to-rate",
        type=str,
        nargs="+",
        help="One or more behavior names to rate. If not specified, rates all available behaviors.",
    )
    rate_config_group.add_argument(
        "--cues-to-rate", type=str, nargs="+", help=argparse.SUPPRESS
    )  # deprecated alias
    rate_config_group.add_argument(
        "--cue-group-config",
        type=str,
        default=None,
        choices=list(CUE_GROUP_CONFIGS.keys()),
        help=(
            "Optional cue grouping: rate several cues per LLM call instead of "
            "one call per cue (see anthro_benchmark.classifier.cue_grouping). "
            "'personal pronoun use' is unaffected (always regex-rated). "
            "Default: unset, i.e. one call per cue as before."
        ),
    )
    rate_config_group.add_argument(
        "--classifier-max-tokens-base",
        type=int,
        default=None,
        help=(
            "Base max_tokens for the classifier LLM's response. Combined with "
            "--classifier-max-tokens-per-cue as base + per_cue * (number of "
            "cues actually asked in that call). Leave both unset for no cap "
            "(default, matches prior behavior). Meant as a circuit breaker "
            "against a runaway/looping generation, not a cost-optimization "
            "lever -- set generously, since a tight cap risks truncating a "
            "response before its Yes/No verdict is emitted."
        ),
    )
    rate_config_group.add_argument(
        "--classifier-max-tokens-per-cue",
        type=int,
        default=None,
        help=(
            "Per-cue max_tokens increment for the classifier LLM, added once "
            "per cue actually asked about in a given call (see "
            "--classifier-max-tokens-base). Matters most with "
            "--cue-group-config, where a call unit's cue count varies by "
            "config and by how --behaviors-to-rate filters it."
        ),
    )
    rate_config_group.add_argument(
        "--num-samples",
        type=int,
        default=1,
        choices=[1, 3],
        help="Number of times to sample rating for each turn per model (1 or 3, default: 1).",
    )

    # Same mechanism/helper (_build_budget_guard) as the generate
    # subcommand -- see that argument group's help text. One BudgetGuard
    # is shared across every cue unit and every classifier model in this
    # run (analogous to generate sharing one guard between the user and
    # target LLMs).
    rate_budget_group = rate_parser.add_argument_group("Budget options")
    rate_budget_group.add_argument(
        "--budget-session-id",
        type=str,
        default=None,
        help="Optional session ID for budget tracking.",
    )
    rate_budget_group.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Hard cap on total classifier LLM calls for the run.",
    )
    rate_budget_group.add_argument(
        "--max-budget-per-session",
        type=float,
        default=None,
        help="Hard cap on estimated spend for the run.",
    )
    rate_budget_group.add_argument(
        "--input-cost-per-1m-tokens",
        type=float,
        default=None,
        help="Prompt-token price used for budget estimation.",
    )
    rate_budget_group.add_argument(
        "--output-cost-per-1m-tokens",
        type=float,
        default=None,
        help="Completion-token price used for budget estimation.",
    )

    rate_concurrency_group = rate_parser.add_argument_group("Concurrency options")
    rate_concurrency_group.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help=(
            "How many turns to rate at once, per (cue unit, classifier "
            "model) combination (default: 1, fully sequential -- identical "
            "to the original behavior). Ratings have no cross-row "
            "dependency (unlike generate's dialogue turns), so this is "
            "typically where concurrency helps most. Runs via "
            "asyncio.to_thread; same overshoot caveat as generate's "
            "--max-concurrency applies if a budget cap is also set."
        ),
    )
    rate_concurrency_group.add_argument(
        "--strict-batch-ordering",
        action="store_true",
        help=(
            "Only relevant with --max-concurrency > 1. Default off: "
            "dispatch all rows in a (cue unit, model) combination "
            "continuously for maximum throughput. Set this to dispatch "
            "in chunks of --max-concurrency instead, one chunk fully "
            "finished before the next starts, at a measured throughput "
            "cost (same order of magnitude as generate's "
            "--strict-batch-ordering). This is what --incremental-save "
            "needs to checkpoint safely under concurrency -- "
            "automatically enabled if --incremental-save is set together "
            "with --max-concurrency > 1."
        ),
    )
    rate_concurrency_group.add_argument(
        "--incremental-save",
        action="store_true",
        help=(
            "Default off: the rated CSV is written once, after the "
            "entire run finishes -- a crash/hang/kill before that point "
            "(e.g. a stuck call during --cue-group-config rating) saves "
            "nothing at all. Set this to rewrite the CSV after each "
            "completed row (sequential) or each completed chunk of rows "
            "(concurrent -- forces --strict-batch-ordering on "
            "automatically) within a (cue unit, model)'s rating, as well "
            "as after each cue unit's columns are fully written. Each "
            "save is atomic (temp file + rename), so an interruption "
            "during the save itself can't corrupt the previous good "
            "checkpoint either. See rate_dialogues()'s docstring in "
            "rating.py for exactly which columns get partial values "
            "written progressively vs. only at full completion."
        ),
    )
    rate_concurrency_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Requires --incremental-save. If --output-rated-csv (or the "
            "generated default path) already exists, merges in its "
            "per-model '{cue}_{model}_final_present' columns -- matched "
            "by (dialogue_id, turn_pair_index) -- and skips re-rating any "
            "(row, cue, model) combination that's already there, reusing "
            "the existing value. Works across a --cue-group-config "
            "change between runs (a cue+model's result means the same "
            "thing regardless of which unit grouped it). Reused rows' "
            "raw-explanation/per-sample columns become a clearly-labeled "
            "placeholder (only the final score is checkpointed "
            "incrementally, not the per-sample detail) -- the final "
            "score itself is exact, never approximated. If the output "
            "path doesn't exist yet, this has no effect -- a normal "
            "fresh run."
        ),
    )

    rate_rate_limit_group = rate_parser.add_argument_group("Rate limiting options")
    rate_rate_limit_group.add_argument(
        "--classifier-min-seconds-between-calls",
        type=float,
        default=None,
        help=(
            "Minimum wait, in seconds, enforced between the START of "
            "consecutive classifier calls -- a flat pacing throttle "
            "(your 'waiting time'), independent of the caps below. Under "
            "--classifier-rate-limit-scope=per-model (the default), each "
            "model in --classifier-model gets its own pacing cadence; "
            "under shared, every model is paced against ONE combined "
            "cadence. Default: unset (no pacing)."
        ),
    )
    rate_rate_limit_group.add_argument(
        "--classifier-max-requests-per-minute",
        type=float,
        default=None,
        help=(
            "Cap on classifier calls in any rolling 60-second window. "
            "Mutually exclusive with --classifier-max-requests-per-second "
            "(same underlying cap, different unit)."
        ),
    )
    rate_rate_limit_group.add_argument(
        "--classifier-max-requests-per-second",
        type=float,
        default=None,
        help=(
            "Same cap as --classifier-max-requests-per-minute, expressed "
            "per second (converted internally as value * 60). Mutually "
            "exclusive with --classifier-max-requests-per-minute."
        ),
    )
    rate_rate_limit_group.add_argument(
        "--classifier-max-tokens-per-minute",
        type=float,
        default=None,
        help=(
            "Cap on total tokens (input+output combined) in any rolling "
            "60-second window, enforced using each call's ACTUAL usage "
            "once it's known -- a single large call can still push the "
            "window over the cap; later calls are then held back to "
            "compensate. Reactive, not a hard preemptive guarantee -- see "
            "RateLimiter's docstring in llm_client.py."
        ),
    )
    rate_rate_limit_group.add_argument(
        "--classifier-rate-limit-scope",
        choices=["shared", "per-model"],
        default="per-model",
        help=(
            "Only matters if at least one cap above is set AND more than "
            "one --classifier-model is given. 'per-model' (default): each "
            "classifier model gets its OWN independent budget at the same "
            "configured limits -- the safer default for rating, since "
            "mixing a free/tightly-limited model with a paid one is "
            "common here, and throttling them together would needlessly "
            "slow the paid model down to the free one's pace. 'shared': "
            "every classifier model is throttled TOGETHER against one "
            "combined budget -- use this only if every model in "
            "--classifier-model genuinely draws on the same provider "
            "account/key and you want their combined call rate bounded "
            "as one pool."
        ),
    )

    rate_parser.set_defaults(func=rate_dialogues_command)

    # Summarize results subcommand
    summarize_parser = subparsers.add_parser(
        "summarize", help="Analyze rated dialogues and generate summaries/plots."
    )

    summarize_parser.add_argument(
        "--rated-csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing rated dialogues (output from 'rate' command).",
    )
    summarize_parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results (plots, summary stats) (default: analysis_results).",
    )
    summarize_parser.add_argument(
        "--first-n-turns",
        type=int,
        default=5,
        help=(
            "In addition to the 'final' results (every rated turn, "
            "whatever length each dialogue reached), also compute the "
            "same statistics restricted to each dialogue's first N "
            "turn-pairs -- for a fixed-length comparison against prior "
            "work that didn't have variable-length dialogues (e.g. from "
            "--stop-on-natural-end). A dialogue that ended earlier than "
            "N turns contributes whatever it has rather than being "
            "excluded, unless --require-min-turns is also set. "
            "Default: 5."
        ),
    )
    summarize_parser.add_argument(
        "--require-min-turns",
        action="store_true",
        default=False,
        help=(
            "Off by default. When set, the first-N-turns results ONLY "
            "include dialogues whose total length is at least "
            "--first-n-turns -- shorter dialogues are excluded from that "
            "view entirely (they're still included in the 'final' "
            "results). Use this for a strict fixed-length comparison "
            "where every included dialogue contributes exactly N turns, "
            "not fewer. Output filenames get a '_strict' suffix when "
            "this is set, so toggling it and re-running doesn't "
            "overwrite the other version's output in the same "
            "--output-dir."
        ),
    )

    summarize_parser.set_defaults(func=summarize_command)

    return parser.parse_args()


def run(args):
    if args.command == "generate":
        generate_dialogues_command(args)
    elif args.command == "rate":
        rate_dialogues_command(args)
    elif args.command == "summarize":
        summarize_command(args)
    else:
        print(f"Unknown command: {args.command}")


def main():
    app.run(run, flags_parser=_parse_flags)


if __name__ == "__main__":
    main()
