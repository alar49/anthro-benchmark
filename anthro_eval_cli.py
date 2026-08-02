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
import re
import traceback

from absl import app
from absl.flags import argparse_flags

from anthro_benchmark.generator import (
    DialogueGenerator,
    LLMGenerationError,
    DEFAULT_USER_SYSTEM_PROMPT,
)
from anthro_benchmark.classifier import run_rating_process
from anthro_benchmark.core.llm_client import BudgetGuard, estimate_cost_from_usage


# helper function for file/column name sanitation
def sanitize_model_name(model_name: str) -> str:
    """Removes characters problematic for filenames/column names."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name)

### EDIT FROM HERE ###

# replace the helper block
def _reasoning_effort_from_args(args):
    if args.reasoning_mode == "off":
        return None
    if args.reasoning_effort:
        return args.reasoning_effort
    if args.reasoning_mode == "on":
        return "medium"  # OpenRouter default when reasoning is enabled
    return None


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
    
### EDITING END ###

def generate_dialogues_command(args):
    print("Starting dialogue generation...")
    print(f"Configuration: {args}")
    print("-" * 30)

    ### EDIT START ###
    
    reasoning_effort = _reasoning_effort_from_args(args)
    reasoning_mode = reasoning_effort is not None

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
    
    if budget_guard is not None:
        user_llm_config["budget_guard"] = budget_guard
        target_llm_config["budget_guard"] = budget_guard
        
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

    reasoning_effort = _reasoning_effort_from_args(args) ### EDITED
    
    try:
        cues_to_rate_arg = (
            args.behaviors_to_rate
            if hasattr(args, "behaviors_to_rate") and args.behaviors_to_rate
            else getattr(args, "cues_to_rate", None)
        )
        output_path = run_rating_process(
            dialogues_csv_path=args.dialogues_csv,
            cues_to_rate=cues_to_rate_arg,
            classifier_models=args.classifier_model,
            classifier_temperature=args.classifier_temperature,
            num_samples=args.num_samples,
            output_rated_csv=getattr(args, "output_rated_csv", None),
            classifier_reasoning_effort=reasoning_effort, ### EDITED
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

        run_analysis(rated_csv_path=args.rated_csv, output_dir=args.output_dir)
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
        "--max-retries",
        type=int,
        default=5,
        help="Maximum transient retry attempts per LLM call.",
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
        "--num-samples",
        type=int,
        default=1,
        choices=[1, 3],
        help="Number of times to sample rating for each turn per model (1 or 3, default: 1).",
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
