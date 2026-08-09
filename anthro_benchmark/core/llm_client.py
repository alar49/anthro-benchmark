# # Copyright 2025 The Anthropomorphism Benchmark Project Authors
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #       https://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# 
# """
# LLM client initialization and management for dialogue generation.
# 
# Alternative Gem_version
# """
# 
# import dataclasses
# import os
# from typing import Optional
# from openai import OpenAI
# 
# 
# @dataclasses.dataclass
# class LLMClient:
#     """Base class for LLM clients."""
# 
#     model: str
#     temperature: float = 0.7
#     # --- NEW: Explicitly define fields so __init__ doesn't crash ---
#     reasoning_mode: bool = False
#     reasoning_effort: Optional[str] = None
# 
#     def generate(self, messages: list, **kwargs) -> str:
#         """
#         Generate a response from the LLM.
# 
#         Args:
#             messages: List of message dictionaries
#             **kwargs: Additional parameters to pass to the LLM
# 
#         Returns:
#             Generated text response
#         """
#         # Route OpenRouter models through the official OpenAI client
#         if self.model.startswith("openrouter/"):
#             # Strip the prefix to get the exact Model ID
#             actual_model = self.model.replace("openrouter/", "")
#             
#             client = OpenAI(
#                 base_url="https://openrouter.ai/api/v1",
#                 api_key=os.environ.get("OPENROUTER_API_KEY")
#             )
#             
#             extra_body = kwargs.pop("extra_body", {})
#             
#             # Use instance variables (self) instead of popping kwargs
#             if self.reasoning_mode:
#                 # OpenRouter standard for exposing reasoning tokens uses an underscore
#                 extra_body["include_reasoning"] = True
#                 
#             if self.reasoning_effort:
#                 # Standard OpenAI-compatible param uses an underscore
#                 kwargs["reasoning_effort"] = self.reasoning_effort
#                 
#             if extra_body:
#                 kwargs["extra_body"] = extra_body
# 
#             # Execute the call
#             response = client.chat.completions.create(
#                 model=actual_model,
#                 messages=messages,
#                 temperature=self.temperature,
#                 **kwargs
#             )
#             
#             # Return the generated text
#             return response.choices[0].message.content
#             
#         # Fallback to litellm for all other providers
#         else:
#             import litellm
#             
#             # Pass reasoning effort to litellm if it's set
#             if self.reasoning_effort:
#                 kwargs["reasoning_effort"] = self.reasoning_effort
# 
#             response = litellm.completion(
#                 model=self.model,
#                 messages=messages,
#                 temperature=self.temperature,
#                 **kwargs
#             )
#             #return response.choices[0].message.content #changing return to...
# 
#             message = response.choices[0].message
#             content = message.content
#             
#             # If you want to append the reasoning to the content:
#             if self.reasoning_mode and hasattr(message, "model_extra") and message.model_extra:
#                 reasoning = message.model_extra.get("reasoning")
#                 if reasoning:
#                     return f"<think>\n{reasoning}\n</think>\n\n{content}"
#                     
#             return content


### NEW VERSION ###

import dataclasses
import logging
import os
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import litellm
try:
    from litellm import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        PermissionDeniedError,
        RateLimitError,
    )
except ImportError:
    from litellm import (
        APIConnectionError,
        APIError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        PermissionDeniedError,
        RateLimitError,
    )
    from openai import APITimeoutError

logger = logging.getLogger(__name__)

ALLOWED_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


class BudgetExceededError(RuntimeError):
    """Raised when the external per-session budget guard is exceeded."""


@dataclasses.dataclass
class BudgetGuard:
    """
    External safety layer.

    - max_iterations: hard cap on the number of LLM calls per session
    - max_budget_per_session: hard cap on spend per session
    - cost_estimator: required when max_budget_per_session is set

    Thread-safe: reserve_call()/record_response() are internally locked so
    this can be shared across concurrent callers (e.g. multiple dialogues
    or rating rows generated in parallel via asyncio.to_thread, which runs
    on real OS threads). Under concurrency the check-then-increment in
    reserve_call() is only atomic *with the lock*; without it, two threads
    could both pass the check before either increments, silently
    exceeding max_iterations. The lock also means the effective ceiling
    under concurrency is "max_iterations, plus however many calls were
    already past their own reserve_call() and in flight" -- not an exact
    global ceiling -- since already-reserved in-flight calls aren't
    retroactively cancelled. That's an inherent property of concurrent
    enforcement, not a bug: keep max_concurrency modest relative to
    max_iterations if you want the overshoot bound to stay small.
    """

    session_id: str
    max_iterations: int = 25
    max_budget_per_session: Optional[float] = None
    cost_estimator: Optional[Callable[[Any], float]] = None
    iterations: int = 0
    spent: float = 0.0
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        if self.max_budget_per_session is not None and self.max_budget_per_session < 0:
            raise ValueError("max_budget_per_session must be >= 0")

        if self.max_budget_per_session is not None and self.cost_estimator is None:
            raise ValueError(
                "max_budget_per_session requires a cost_estimator for reliable dollar enforcement."
            )

    def reserve_call(self) -> None:
        """
        Reserve one actual API call before sending it.
        This makes retries count too.
        """
        with self._lock:
            if self.iterations >= self.max_iterations:
                raise BudgetExceededError(
                    f"Max iterations exceeded for session '{self.session_id}' "
                    f"({self.iterations}/{self.max_iterations})."
                )

            if self.max_budget_per_session is not None and self.spent >= self.max_budget_per_session:
                raise BudgetExceededError(
                    f"Max budget exceeded for session '{self.session_id}' "
                    f"(${self.spent:.6f}/${self.max_budget_per_session:.6f})."
                )

            self.iterations += 1

    def record_response(self, response: Any) -> None:
        """
        Add the estimated cost of a successful response.
        """
        if self.cost_estimator is None:
            return

        spend = float(self.cost_estimator(response))
        if spend < 0:
            raise ValueError("cost_estimator returned a negative spend estimate.")

        with self._lock:
            self.spent += spend

            if self.max_budget_per_session is not None and self.spent > self.max_budget_per_session:
                raise BudgetExceededError(
                    f"Max budget exceeded for session '{self.session_id}' "
                    f"(${self.spent:.6f}/${self.max_budget_per_session:.6f})."
                )


def estimate_cost_from_usage(
    response: Any,
    input_cost_per_1m_tokens: float,
    output_cost_per_1m_tokens: float,
    *,
    strict: bool = True,
) -> float:
    """
    Generic cost estimator.

    Supply your actual model/provider pricing here.
    If strict=True and the response has no usage block, we fail fast
    rather than silently undercounting spend.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        if strict:
            raise RuntimeError(
                "Response does not expose usage; cannot enforce max_budget_per_session safely."
            )
        return 0.0

    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", None)

    output_tokens = getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", None)

    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)

    return (
        (input_tokens / 1_000_000.0) * float(input_cost_per_1m_tokens)
        + (output_tokens / 1_000_000.0) * float(output_cost_per_1m_tokens)
    )


def _merge_metadata(*parts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for part in parts:
        if isinstance(part, dict):
            metadata.update(part)
    return metadata


def _stringify_reasoning_content(reasoning_content: Any) -> str:
    if reasoning_content is None:
        return ""

    if isinstance(reasoning_content, str):
        return reasoning_content.strip()

    if isinstance(reasoning_content, list):
        chunks: List[str] = []
        for item in reasoning_content:
            if isinstance(item, str):
                chunks.append(item)
                continue

            if isinstance(item, dict):
                for key in ("text", "reasoning", "content", "thinking"):
                    value = item.get(key)
                    if value:
                        chunks.append(str(value))
                        break
                continue

            for attr in ("text", "reasoning", "content", "thinking"):
                value = getattr(item, attr, None)
                if value:
                    chunks.append(str(value))
                    break

        return "\n".join(chunks).strip()

    return str(reasoning_content).strip()


def _extract_reasoning_content(message: Any) -> str:
    reasoning_content = getattr(message, "reasoning_content", None)

    if reasoning_content is None:
        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            reasoning_content = (
                model_extra.get("reasoning_content")
                or model_extra.get("reasoning")
                or model_extra.get("thinking_blocks")
            )

    if reasoning_content is None:
        reasoning_content = getattr(message, "thinking_blocks", None)

    return _stringify_reasoning_content(reasoning_content)


def _extract_reasoning_token_count(usage: Any) -> Optional[int]:
    """
    Best-effort extraction of a reasoning/thinking token count from a
    response's `usage` block, across the different shapes providers use.

    Returns None if no such field is present at all -- that's meaningful on
    its own, since it means this provider/response simply doesn't report
    reasoning token spend, as opposed to reporting zero.
    """
    if usage is None:
        return None

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    # OpenAI-style nested shape (also what LiteLLM normalizes o-series/
    # reasoning-model usage into): usage.completion_tokens_details.reasoning_tokens
    details = _get(usage, "completion_tokens_details")
    if details is not None:
        value = _get(details, "reasoning_tokens")
        if value is not None:
            return int(value)

    # Some providers (seen via OpenRouter passthroughs) put it directly
    # on the usage object instead of nesting it.
    for key in ("reasoning_tokens", "thinking_tokens"):
        value = _get(usage, key)
        if value is not None:
            return int(value)

    # LiteLLM sometimes stashes provider-specific extras here.
    model_extra = _get(usage, "model_extra")
    if isinstance(model_extra, dict):
        for key in ("reasoning_tokens", "thinking_tokens"):
            if model_extra.get(key) is not None:
                return int(model_extra[key])

    return None


# --- Provider-aware API key resolution -------------------------------------
#
# LiteLLM identifies a model's provider either from an explicit
# "<provider>/<model>" prefix (e.g. "anthropic/claude-3-5-sonnet-20241022")
# or by recognizing the model name itself even with no prefix at all (e.g.
# "claude-3-5-sonnet-20241022" and "gemini-1.5-flash" are both valid on
# their own). Each provider expects its own environment variable for
# credentials. This maps a model string to the ordered list of env vars to
# check, so Gemini/Claude/Mistral models are resolved the same way
# OpenAI/OpenRouter ones already were.
PROVIDER_API_KEY_ENV_VARS: Dict[str, List[str]] = {
    "openrouter": ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "vertex_ai": ["GOOGLE_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
}

# Substring fallbacks for prefix-less model names (e.g. "claude-3-5-sonnet",
# "gemini-1.5-flash", "mistral-large-latest", "gpt-4o"), checked in order if
# the "<prefix>/..." form doesn't match a known provider above. Order
# matters only in that these are checked before giving up, not relative to
# each other (the substrings don't overlap).
_PROVIDER_NAME_HINTS: List[Tuple[str, str]] = [
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("gpt", "openai"),
]


def _infer_provider(model: str) -> Optional[str]:
    """Best-effort provider name for `model`, used only to choose which
    environment variable(s) to check for an API key. Returns None if no
    provider could be inferred at all -- LiteLLM itself recognizes far more
    providers/env-var names than the handful mapped above (Bedrock, Vertex
    service accounts, Azure, HuggingFace, Cohere, ...), so callers should
    fall through to LiteLLM's own resolution rather than treating None as
    an error."""
    prefix = model.split("/", 1)[0].lower() if "/" in model else ""
    if prefix in PROVIDER_API_KEY_ENV_VARS:
        return prefix

    lower_model = model.lower()
    for hint, provider in _PROVIDER_NAME_HINTS:
        if hint in lower_model:
            return provider

    return None


def resolve_api_key_env_vars(model: str) -> List[str]:
    """Ordered list of environment variable names to check for `model`.
    Empty list if a provider couldn't be inferred at all (see
    _infer_provider) -- that's meaningful on its own (nothing to check),
    as distinct from a recognized provider whose env var(s) simply aren't
    set."""
    provider = _infer_provider(model)
    if provider is None:
        return []
    return PROVIDER_API_KEY_ENV_VARS[provider]


class LLMClient:
    """
    LiteLLM-first client.

    Design goals:
    - route OpenRouter through LiteLLM too
    - keep temperature caller-controlled
    - keep reasoning visible to the caller
    - allow pinning which OpenRouter backend(s) serve a model, since
      OpenRouter load-balances across providers by default and different
      providers for the same model slug can silently support different
      request parameters (e.g. one honors `reasoning`, another drops it)
    - surface which OpenRouter backend actually served each call (when
      available) so that kind of silent mismatch shows up in the logs
      instead of requiring a manual OpenRouter dashboard check
    - retry only transient failures
    - fail fast on config/programming mistakes
    - keep unknown-exception retries very limited
    - avoid hidden retry loops by forcing num_retries=0 here
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        reasoning_mode: bool = False,
        reasoning_effort: Optional[str] = None,
        openrouter_provider: Optional[Dict[str, Any]] = None,
        openrouter_router_metadata: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0,
        budget_guard: Optional[BudgetGuard] = None,
        **extra_config: Any,
    ):
        self.raw_model = model
        self.temperature = temperature
        self.reasoning_mode = reasoning_mode
        self.reasoning_effort = reasoning_effort
        self.openrouter_provider = openrouter_provider
        self.openrouter_router_metadata = openrouter_router_metadata
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.extra_config = extra_config
        self.budget_guard = budget_guard

        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if self.initial_backoff <= 0:
            raise ValueError("initial_backoff must be > 0")
        if self.max_backoff < self.initial_backoff:
            raise ValueError("max_backoff must be >= initial_backoff")

        self.model = model

        if model.startswith("openrouter/"):
            resolved_base_url = base_url or "https://openrouter.ai/api/v1"
        else:
            resolved_base_url = base_url

        # Try an explicit api_key argument first, then whichever
        # provider-specific environment variable(s) apply to this model
        # (see PROVIDER_API_KEY_ENV_VARS / _infer_provider above). This is
        # what makes GOOGLE_API_KEY/GEMINI_API_KEY, ANTHROPIC_API_KEY, and
        # MISTRAL_API_KEY actually work for direct (non-OpenRouter) Gemini/
        # Claude/Mistral models -- previously only OPENAI_API_KEY (and, for
        # "openrouter/..." models, OPENROUTER_API_KEY) were ever checked
        # here, regardless of which provider the model string named.
        candidate_env_vars = resolve_api_key_env_vars(model)
        resolved_api_key = api_key
        if resolved_api_key is None:
            for env_var in candidate_env_vars:
                value = os.environ.get(env_var)
                if value:
                    resolved_api_key = value
                    break

        if resolved_api_key is None:
            # Don't hard-fail here. If the provider couldn't be inferred at
            # all, or none of its candidate env vars are set, fall through
            # and let LiteLLM's own credential resolution take over --
            # LiteLLM recognizes many more providers/env-var names than the
            # handful mapped above (Bedrock, Vertex service accounts,
            # Azure, HuggingFace, Cohere, a bare litellm.api_key, ...), so
            # refusing to even try here would be more restrictive than
            # useful. If LiteLLM also can't find credentials, the call
            # raises its own AuthenticationError with a provider-specific
            # message (handled, not retried, in generate() below) --
            # clearer than a generic error from this constructor would be.
            if candidate_env_vars:
                logger.warning(
                    "No API key found for model '%s'. Checked: %s (and the "
                    "api_key argument). Falling back to LiteLLM's own "
                    "credential resolution; the call will fail with its own "
                    "error if that also finds nothing.",
                    model,
                    ", ".join(candidate_env_vars),
                )
            else:
                logger.warning(
                    "Could not infer a provider for model '%s' to pick a "
                    "specific API key environment variable. Falling back to "
                    "LiteLLM's own credential resolution.",
                    model,
                )

        if self.openrouter_provider and not self.model.startswith("openrouter/"):
            raise ValueError(
                f"openrouter_provider={self.openrouter_provider!r} was set but "
                f"model={self.model!r} is not an 'openrouter/...' model. "
                "OpenRouter's provider-routing object (order/allow_fallbacks/"
                "require_parameters/etc.) only has an effect on requests that "
                "actually go to OpenRouter, so this is almost certainly a "
                "config mistake rather than a no-op you want."
            )

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url

        # Keep the old attribute name while using LiteLLM as the primary layer.
        self.client = litellm

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        return_reasoning: bool = False,
        **kwargs,
    ) -> Union[str, Tuple[str, str]]:
        """
        Send messages via LiteLLM and return the model's reply.

        By default this returns ONLY the final answer as a plain string,
        even when reasoning/thinking mode is enabled. The reasoning trace
        (if any) is never concatenated into that string. This matters
        because the returned value is reused verbatim as conversation
        history, CSV output, and classifier-prompt input elsewhere in this
        codebase -- none of which are designed to parse a "<think>...</think>"
        blob, and feeding it to them corrupts history/echo behavior and
        rating results.

        Pass return_reasoning=True to also get the reasoning trace (e.g.
        for logging/inspection): the return value is then a
        (content, reasoning_text) tuple, where reasoning_text is "" if the
        provider returned none.
        """

        budget_guard = kwargs.pop("budget_guard", self.budget_guard)

        runtime_metadata = kwargs.pop("metadata", None)
        runtime_reasoning = kwargs.pop("reasoning", None)

        reasoning_mode = kwargs.pop("reasoning_mode", self.reasoning_mode)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)
        openrouter_provider = kwargs.pop("openrouter_provider", self.openrouter_provider)
        openrouter_router_metadata = kwargs.pop(
            "openrouter_router_metadata", self.openrouter_router_metadata
        )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "num_retries": 0,  # centralize retry behavior here; no hidden retry loops
        }
        if self.api_key:
            # Only set explicitly when we actually resolved one (see
            # __init__) -- omitting the key entirely, rather than passing
            # None, lets LiteLLM fall back to its own credential
            # resolution instead of e.g. sending a literal "None".
            payload["api_key"] = self.api_key

        if self.base_url:
            payload["api_base"] = self.base_url

        if self.extra_config:
            payload.update(self.extra_config)

        if kwargs:
            payload.update(kwargs)

        if openrouter_provider:
            # OpenRouter's provider-routing object (order/allow_fallbacks/
            # require_parameters/etc., see
            # https://openrouter.ai/docs/guides/routing/provider-selection)
            # has to travel inside `extra_body`, not as a bare top-level
            # `provider=` kwarg. LiteLLM's OpenAI-compatible layer only
            # recognizes a fixed set of top-level params; passing
            # provider-specific fields any other way has historically been
            # silently dropped or rejected outright depending on the LiteLLM
            # version (see https://github.com/BerriAI/litellm/issues/6857).
            # `extra_body` is the one channel LiteLLM explicitly merges
            # verbatim into the JSON body it sends to OpenRouter.
            extra_body = dict(payload.get("extra_body") or {})
            if "provider" in extra_body:
                raise ValueError(
                    "Both openrouter_provider and an extra_body['provider'] "
                    "(passed via extra_config or generate() kwargs) were set "
                    "for the same call. Set the provider-routing object in "
                    "exactly one place to avoid ambiguity about which wins."
                )
            extra_body["provider"] = openrouter_provider
            payload["extra_body"] = extra_body

        if self.model.startswith("openrouter/") and openrouter_router_metadata:
            # Ask OpenRouter to include routing metadata (which underlying
            # provider actually served this call, whether it fell back to
            # another one, etc.) on the response, so a provider silently
            # mis-serving a call -- the exact Google AI Studio/Darkbloom
            # reasoning issue that motivated openrouter_provider above --
            # is visible in the log line below without needing OpenRouter's
            # dashboard. This is an EXPERIMENTAL OpenRouter feature (see
            # https://openrouter.ai/docs/guides/features/router-metadata):
            # the response shape may change without notice. OpenRouter's own
            # docs are inconsistent about whether "X-OpenRouter-Experimental-
            # Metadata" or "X-OpenRouter-Metadata" is the current header
            # name (one page calls the former current, another calls it
            # legacy), so both are sent; an unrecognized header is harmless.
            extra_headers = dict(payload.get("extra_headers") or {})
            extra_headers.setdefault("X-OpenRouter-Experimental-Metadata", "enabled")
            extra_headers.setdefault("X-OpenRouter-Metadata", "enabled")
            payload["extra_headers"] = extra_headers

        if budget_guard is not None:
            payload["metadata"] = _merge_metadata(
                payload.get("metadata"),
                runtime_metadata,
                {"session_id": budget_guard.session_id},
            )
        elif runtime_metadata is not None:
            payload["metadata"] = _merge_metadata(payload.get("metadata"), runtime_metadata)

        if runtime_reasoning is not None:
            payload["reasoning"] = runtime_reasoning
        elif reasoning_mode or reasoning_effort:
            reasoning: Dict[str, Any] = {
                "enabled": True,
                "exclude": False,  # keep reasoning tokens in the response for exploration
            }

            if reasoning_effort:
                if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
                    raise ValueError(
                        f"Unsupported reasoning_effort={reasoning_effort!r}. "
                        f"Supported values include: {sorted(ALLOWED_REASONING_EFFORTS)}"
                    )
                reasoning["effort"] = reasoning_effort

            # Optional cap if you later want to limit thinking spend:
            # reasoning["max_tokens"] = 1024

            payload["reasoning"] = reasoning

        attempt = 0
        unknown_exception_attempts = 0
        current_backoff = self.initial_backoff

        while attempt < self.max_retries:
            if budget_guard is not None:
                budget_guard.reserve_call()

            attempt += 1
            try:
                response = self.client.completion(**payload)

                if budget_guard is not None:
                    budget_guard.record_response(response)

                if not getattr(response, "choices", None):
                    return ("", "") if return_reasoning else ""

                choice = response.choices[0]
                message = getattr(choice, "message", None)
                if message is None:
                    return ("", "") if return_reasoning else ""

                content = getattr(message, "content", None) or ""
                reasoning_text = _extract_reasoning_content(message)

                usage = getattr(response, "usage", None)
                reasoning_token_count = _extract_reasoning_token_count(usage)

                # See the extra_headers block above: when present, this is
                # OpenRouter's own account of which provider served the
                # call, whether it fell back, etc. -- not something we
                # infer indirectly from token counts.
                openrouter_metadata = getattr(response, "openrouter_metadata", None)
                served_by_provider = None
                if isinstance(openrouter_metadata, dict):
                    endpoints = openrouter_metadata.get("endpoints") or {}
                    for endpoint in endpoints.get("available") or []:
                        if isinstance(endpoint, dict) and endpoint.get("selected"):
                            served_by_provider = endpoint.get("provider")
                            break

                logger.info(
                    "LLM call usage | model=%s reasoning_requested=%s "
                    "reasoning_text_extracted=%s reasoning_tokens_reported=%s "
                    "prompt_tokens=%s completion_tokens=%s total_tokens=%s "
                    "openrouter_provider_requested=%s openrouter_served_by=%s",
                    self.model,
                    bool(reasoning_mode or reasoning_effort or runtime_reasoning is not None),
                    bool(reasoning_text),
                    reasoning_token_count,
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                    getattr(usage, "total_tokens", None),
                    openrouter_provider or None,
                    served_by_provider,
                )
                if openrouter_metadata is not None:
                    # Full router metadata (attempts, fallbacks, pipeline
                    # stages, etc.) at DEBUG level -- verbose, but this is
                    # exactly what you'd want when actually chasing down a
                    # routing problem rather than just confirming one exists.
                    logger.debug(
                        "LLM call OpenRouter routing metadata | model=%s metadata=%s",
                        self.model,
                        openrouter_metadata,
                    )

                if return_reasoning:
                    return content, reasoning_text

                return content

            except BadRequestError as e:
                logger.error(
                    "Permanent BadRequestError (Attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                raise

            except (AuthenticationError, PermissionDeniedError) as e:
                logger.error(
                    "Permanent Auth/Permission Error (Attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                raise

            except (
                RateLimitError,
                InternalServerError,
                APIConnectionError,
                APITimeoutError,
                APIError,
            ) as e:
                status_code = getattr(e, "status_code", None)

                # Fail fast on general client errors, except 429.
                if status_code and 400 <= status_code < 500 and status_code != 429:
                    logger.error("Non-retryable client error status %s: %s", status_code, e)
                    raise

                if attempt >= self.max_retries:
                    logger.error(
                        "Max retry attempts (%s) exhausted for model '%s'. Last error: %s",
                        self.max_retries,
                        self.model,
                        e,
                    )
                    raise

                retry_after_header = None
                if hasattr(e, "response") and getattr(e, "response", None) is not None:
                    retry_after_header = e.response.headers.get("Retry-After")
                elif hasattr(e, "headers") and getattr(e, "headers", None) is not None:
                    retry_after_header = e.headers.get("Retry-After")

                if retry_after_header:
                    try:
                        sleep_time = float(retry_after_header)
                        logger.info(
                            "Honoring Retry-After header: waiting %s seconds.",
                            sleep_time,
                        )
                    except ValueError:
                        sleep_time = min(
                            current_backoff * random.uniform(0.75, 1.25),
                            self.max_backoff,
                        )
                        current_backoff *= 2.0
                else:
                    sleep_time = min(
                        current_backoff * random.uniform(0.75, 1.25),
                        self.max_backoff,
                    )
                    current_backoff *= 2.0

                logger.warning(
                    "Transient network/provider error (%s: %s). "
                    "Retrying attempt %s/%s in %.2f seconds...",
                    type(e).__name__,
                    e,
                    attempt,
                    self.max_retries,
                    sleep_time,
                )
                time.sleep(sleep_time)

            except Exception as e:
                # Very limited fallback for truly unexpected exceptions.
                unknown_exception_attempts += 1
                if unknown_exception_attempts > 1 or attempt >= self.max_retries:
                    raise

                sleep_time = min(
                    current_backoff * random.uniform(0.75, 1.25),
                    self.max_backoff,
                )
                current_backoff *= 2.0

                logger.warning(
                    "Unexpected exception (%s: %s). One limited retry in %.2f seconds...",
                    type(e).__name__,
                    e,
                    sleep_time,
                )
                time.sleep(sleep_time)

        raise RuntimeError(
            f"Failed to generate completion for model '{self.model}' after {self.max_retries} attempts."
        )


# Example of a real budget setup outside the class:
#
# budget_guard = BudgetGuard(
#     session_id="session-001",
#     max_iterations=25,
#     max_budget_per_session=5.00,
#     cost_estimator=lambda response: estimate_cost_from_usage(
#         response,
#         input_cost_per_1m_tokens=0.50,   # fill in your real model pricing
#         output_cost_per_1m_tokens=1.50,  # fill in your real model pricing
#     ),
# )
#
# client = LLMClient(
#     model="openrouter/anthropic/claude-3.7-sonnet",
#     temperature=0.4,                 # you control this
#     reasoning_mode=True,
#     reasoning_effort="medium",       # documented values like low/medium/high
#     budget_guard=budget_guard,
#     # Pin routing to a specific backend, in priority order, and refuse to
#     # silently fall back to one that might not honor `reasoning` at all
#     # (see https://openrouter.ai/docs/guides/routing/provider-selection).
#     # Only valid for "openrouter/..." models.
#     openrouter_provider={
#         "order": ["Google AI Studio"],
#         "allow_fallbacks": False,
#         "require_parameters": True,
#     },
#     # Default True for "openrouter/..." models: asks OpenRouter to report
#     # which backend actually served each call (logged at INFO as
#     # openrouter_served_by=..., full detail at DEBUG). Experimental on
#     # OpenRouter's side -- see
#     # https://openrouter.ai/docs/guides/features/router-metadata -- so set
#     # openrouter_router_metadata=False if it ever misbehaves.
# )
#
# answer = client.generate([
#     {"role": "user", "content": "Explain the problem step by step."}
# ])
# print(answer)
