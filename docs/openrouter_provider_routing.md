# OpenRouter provider routing

This guide covers a set of flags added to `anthro-eval generate` and
`anthro-eval rate` that control **which underlying provider OpenRouter uses**
to serve a model, plus a related observability feature that reports which
provider actually served each call. Both exist because of a real bug found
while running this benchmark, described below.

- [The problem this solves](#the-problem-this-solves)
- [Background: how OpenRouter provider routing works](#background-how-openrouter-provider-routing-works)
- [The flags](#the-flags)
  - [`--*-openrouter-provider-order`](#--openrouter-provider-order)
  - [`--*-openrouter-no-fallbacks`](#--openrouter-no-fallbacks)
  - [`--*-openrouter-require-parameters`](#--openrouter-require-parameters)
- [Full flag reference](#full-flag-reference)
- [Worked example: the bug that motivated this](#worked-example-the-bug-that-motivated-this)
- [Observability: seeing which provider served each call](#observability-seeing-which-provider-served-each-call)
- [Using this from Python directly](#using-this-from-python-directly)
- [Common mistakes](#common-mistakes)
- [Caveats and what hasn't been tested](#caveats-and-what-hasnt-been-tested)
- [References](#references)

## The problem this solves

OpenRouter doesn't run models itself; a single model slug (e.g.
`google/gemma-3-27b-it:free`) can be served by several different backend
providers, and OpenRouter load-balances requests across them by default.
That would be invisible and harmless if every provider behaved identically
-- but they don't. In this project, running the *same* target-model config
with `reasoning_mode` enabled produced full reasoning tokens on the first
call of a run and then **silently zero reasoning tokens on every call after
it**, with no error. The cause: OpenRouter was routing the first call to a
backend that honored the `reasoning` request parameter (Google AI Studio),
then falling back to a different backend for later calls (in this case,
"Darkbloom") that accepted the request but simply ignored `reasoning`
instead of rejecting it. Nothing crashed. Nothing warned. The run just quietly
stopped measuring what it thought it was measuring.

The flags below let you pin a model to a specific provider (or ordered list
of providers), refuse to silently fall back to an unlisted one, and/or tell
OpenRouter to only route to providers that support every parameter you
actually sent -- so a mismatch like this fails loudly or falls back visibly
instead of returning a "successful" response with the wrong data in it.

## Background: how OpenRouter provider routing works

Every request OpenRouter receives can carry an optional `provider` object
that controls routing. Full details are in [OpenRouter's own
docs](https://openrouter.ai/docs/guides/routing/provider-selection); the
fields this project exposes are:

| OpenRouter field | What it does |
| --- | --- |
| `order` | An ordered list of provider names to try, e.g. `["Google AI Studio", "DeepInfra"]`. OpenRouter tries them in that order. |
| `allow_fallbacks` | Whether OpenRouter may use a provider **not** in `order` if all listed ones are unavailable. Defaults to `true`. |
| `require_parameters` | If `true`, OpenRouter excludes any provider that doesn't support every parameter in your request (e.g. `reasoning`), instead of routing to it and having it ignore the unsupported one. |

These three are exposed as CLI flags. OpenRouter's `provider` object has
several other fields (`sort`, `ignore`, `only`, `data_collection`,
`quantizations`, `max_price`, ...) that aren't wired up as dedicated flags,
but if you're calling `LLMClient` directly from Python rather than through
the CLI, you can pass any of them -- see
[Using this from Python directly](#using-this-from-python-directly).

## The flags

Three flags exist for **each** LLM role that can talk to OpenRouter:

- `generate`: `--user-llm-*` and `--target-llm-*`
- `rate`: `--classifier-*` (applies identically to every model passed to
  `--classifier-model`)

They're independent per role. Pinning the target model's provider doesn't
touch the user model's, and vice versa.

### `--*-openrouter-provider-order`

Takes one or more provider names, in priority order:

```bash
--target-llm-openrouter-provider-order "Google AI Studio"
```

```bash
--target-llm-openrouter-provider-order "Google AI Studio" "DeepInfra"
```

This alone does **not** forbid other providers -- it just says "prefer
these, in this order." If none of the listed providers are available,
OpenRouter still falls back to whatever else serves the model, unless you
also set `--*-openrouter-no-fallbacks`.

Only meaningful when the corresponding model flag
(`--user-llm-model` / `--target-llm-model` / `--classifier-model`) is an
`openrouter/...` model. Setting it on anything else is an error --
`LLMClient` raises immediately rather than silently doing nothing (see
[Common mistakes](#common-mistakes)).

Provider names are OpenRouter's own display names / slugs (e.g. "Google AI
Studio", "DeepInfra", "Together"). Check the model's page on
[openrouter.ai/models](https://openrouter.ai/models) to see which providers
actually serve it and what they're called there.

### `--*-openrouter-no-fallbacks`

A plain switch (no value). Disables fallback to any provider **not** listed
in `--*-openrouter-provider-order`:

```bash
--target-llm-openrouter-provider-order "Google AI Studio" \
--target-llm-openrouter-no-fallbacks
```

With both flags set, if Google AI Studio is unavailable, the call fails
instead of silently going to another provider. That's the point: an
explicit, loud failure is far more useful than a quiet fallback to a
provider you haven't verified behaves the way you expect.

Setting `--*-openrouter-no-fallbacks` **without** `--*-openrouter-provider-order`
is almost always a mistake -- it disables fallback from whatever single
provider OpenRouter's own default (price-based) selection would have picked,
rather than pinning to a provider list you've chosen. The CLI prints a
warning if you do this, but doesn't stop you (there might be a real use case
for "give me your best-priced provider, but fail rather than fall back if
it's down").

### `--*-openrouter-require-parameters`

A plain switch. Tells OpenRouter to exclude, from routing consideration,
any provider that doesn't support every parameter present in the request:

```bash
--target-llm-openrouter-require-parameters
```

This is the belt-and-suspenders complement to pinning a provider by name.
Even with no `--*-openrouter-provider-order` set at all, this flag alone
would have caught the original bug: OpenRouter would have excluded
"Darkbloom" from consideration for any request that included `reasoning`,
since Darkbloom doesn't honor it, rather than routing there and getting back
a response with `reasoning_tokens: 0`.

It's most useful for the **target** model when `--reasoning-mode on` is set,
since `reasoning` is the parameter most likely to silently vary in support
across providers. It's provided for the user/classifier roles too, for
symmetry, but those roles never request reasoning in this codebase, so it's
unlikely to change anything there.

## Full flag reference

**`generate` subcommand:**

| Flag | Applies to | Type |
| --- | --- | --- |
| `--user-llm-openrouter-provider-order PROVIDER [PROVIDER ...]` | User LLM | ordered list |
| `--user-llm-openrouter-no-fallbacks` | User LLM | switch |
| `--user-llm-openrouter-require-parameters` | User LLM | switch |
| `--target-llm-openrouter-provider-order PROVIDER [PROVIDER ...]` | Target LLM | ordered list |
| `--target-llm-openrouter-no-fallbacks` | Target LLM | switch |
| `--target-llm-openrouter-require-parameters` | Target LLM | switch |

**`rate` subcommand:**

| Flag | Applies to | Type |
| --- | --- | --- |
| `--classifier-openrouter-provider-order PROVIDER [PROVIDER ...]` | every `--classifier-model` | ordered list |
| `--classifier-openrouter-no-fallbacks` | every `--classifier-model` | switch |
| `--classifier-openrouter-require-parameters` | every `--classifier-model` | switch |

None of these have effect unless the relevant model is an `openrouter/...`
model; setting them on a non-OpenRouter model raises an error rather than
being ignored.

## Worked example: the bug that motivated this

This mirrors the actual scenario that surfaced the issue: `google/gemma-3-27b-it:free`
on OpenRouter is served by both Google AI Studio and a second provider
("Darkbloom") that doesn't support `reasoning`.

Pin the target model to Google AI Studio only, and belt-and-suspenders it
with `require_parameters` in case the provider name ever changes or a new
non-reasoning-capable provider gets added later:

```bash
anthro-eval generate \
  --user-llm-model "openrouter/google/gemma-3-27b-it:free" \
  --target-llm-model "openrouter/google/gemma-3-27b-it:free" \
  --reasoning-mode on \
  --reasoning-effort medium \
  --target-llm-openrouter-provider-order "Google AI Studio" \
  --target-llm-openrouter-no-fallbacks \
  --target-llm-openrouter-require-parameters \
  --prompt-category-name "internal states" \
  --num-dialogues 10 \
  --output-dir generated_dialogues
```

Note that `--user-llm-openrouter-*` isn't set here at all: the user LLM
never requests reasoning in this codebase, so it doesn't matter which
provider serves it, and it's fine to let OpenRouter load-balance it as
usual (which is also cheaper/faster, since you're not restricting it to a
single backend).

If you instead want to rate an existing dialogues CSV with the same model
as a classifier, and want the same guarantee:

```bash
anthro-eval rate \
  --dialogues-csv generated_dialogues/your_dialogue_file.csv \
  --classifier-model "openrouter/google/gemma-3-27b-it:free" \
  --classifier-openrouter-provider-order "Google AI Studio" \
  --classifier-openrouter-no-fallbacks \
  --behaviors-to-rate "empathy" "desires"
```

## Observability: seeing which provider served each call

Independent of the flags above, `LLMClient` now asks OpenRouter to report,
on every response, which provider actually served the call. This is on by
default for any `openrouter/...` model (no flag needed) and shows up in the
existing per-call log line:

```
LLM call usage | model=openrouter/google/gemma-3-27b-it:free reasoning_requested=True
reasoning_text_extracted=True reasoning_tokens_reported=847
prompt_tokens=210 completion_tokens=340 total_tokens=550
openrouter_provider_requested={'order': ['Google AI Studio'], 'allow_fallbacks': False}
openrouter_served_by=Google AI Studio
```

`openrouter_served_by` is the actual provider that handled that specific
call, straight from OpenRouter -- not inferred from token counts. This is
exactly the piece of information that would have made the original bug
obvious on the very first affected call, instead of requiring a trip to
OpenRouter's Activity dashboard after the fact.

Enabling more detail: the full routing metadata OpenRouter returns
(routing strategy, fallback attempts, pipeline stages like guardrails or
context compression, etc.) is logged at `DEBUG` level under the same
logger (`anthro_benchmark.core.llm_client`), since it's fairly verbose:

```
LLM call OpenRouter routing metadata | model=openrouter/google/gemma-3-27b-it:free
metadata={'requested': 'google/gemma-3-27b-it:free', 'strategy': 'direct',
'attempt': 1, 'endpoints': {'available': [{'provider': 'Google AI Studio',
'selected': True}, {'provider': 'Darkbloom', 'selected': False}]}}
```

To see it, raise the log level for that logger, e.g. in Python before
running a generation/rating job:

```python
import logging
logging.getLogger("anthro_benchmark.core.llm_client").setLevel(logging.DEBUG)
```

**This feature is opt-in on OpenRouter's side and marked experimental by
OpenRouter.** OpenRouter's docs describe the response shape as unstable --
fields may be added, renamed, or removed without notice -- so treat
`openrouter_served_by` and the DEBUG-level metadata as best-effort debugging
aids, not something to build downstream logic on top of. If it ever stops
showing up or looks wrong, that's OpenRouter's side changing, not
necessarily a bug in this codebase.

If you ever want to turn it off (e.g. if OpenRouter changes something and it
starts causing problems), it's a Python-level constructor flag rather than a
CLI flag right now: `LLMClient(..., openrouter_router_metadata=False)`. There's
no CLI switch for it since there was no reason to expect anyone would want
it off, but this is easy to add if that changes.

## Using this from Python directly

If you're scripting against `LLMClient` directly rather than through the
CLI, `openrouter_provider` accepts the raw OpenRouter `provider` object, so
any field OpenRouter supports works, not just the three the CLI exposes:

```python
from anthro_benchmark.core.llm_client import LLMClient

client = LLMClient(
    model="openrouter/google/gemma-3-27b-it:free",
    reasoning_mode=True,
    reasoning_effort="medium",
    openrouter_provider={
        "order": ["Google AI Studio"],
        "allow_fallbacks": False,
        "require_parameters": True,
        # any other OpenRouter provider field also works here, e.g.:
        # "sort": "throughput",
        # "data_collection": "deny",
    },
)
```

`openrouter_provider` and `openrouter_router_metadata` can also be
overridden per call rather than fixed at construction time:

```python
client.generate(messages, openrouter_provider={"order": ["DeepInfra"]})
client.generate(messages, openrouter_router_metadata=False)
```

## Common mistakes

- **Setting a provider-routing flag on a non-OpenRouter model.** `LLMClient`
  raises a `ValueError` immediately rather than silently ignoring it, since
  a routing object that quietly does nothing is far more confusing than a
  loud error at startup.
- **Mixing OpenRouter and non-OpenRouter models in one `--classifier-model`
  list, with `--classifier-openrouter-*` flags set.** These flags apply
  identically to *every* model in that list. If some are `openrouter/...`
  and some aren't, the non-OpenRouter one(s) will raise the same
  `ValueError` above. Either don't mix, or don't set these flags when you
  do.
- **`--*-openrouter-no-fallbacks` without `--*-openrouter-provider-order`.**
  Valid, but almost never what you want -- see
  [the flag's own section](#--openrouter-no-fallbacks) above. The CLI warns
  when it detects this.
- **Setting both `openrouter_provider` and a hand-rolled
  `extra_body={"provider": ...}`** (only possible via direct Python/`extra_config`
  use, not through the CLI). `LLMClient` raises rather than picking a winner
  silently -- set the provider-routing object in exactly one place.
- **Assuming a provider name from the OpenRouter website works verbatim.**
  Double-check the exact name/slug on the model's OpenRouter page; a typo'd
  provider name is simply never matched, which (with `--no-fallbacks` set)
  looks identical to "provider unavailable" from the outside.

## Caveats and what hasn't been tested

Everything above was verified against the actual installed `litellm`
version's internals (request-body construction, response-object field
passthrough) and with mocked end-to-end tests through the CLI, but **not
against a live OpenRouter call** -- there was no live network access
available while implementing this. Specifically un-tested against the real
API:

- Whether OpenRouter accepts both `X-OpenRouter-Metadata` and
  `X-OpenRouter-Experimental-Metadata` gracefully when both are sent (the
  implementation sends both as a hedge, since OpenRouter's own docs
  disagree on which one is current vs. legacy).
- The exact real-world shape of `openrouter_metadata` on a genuine response
  for the models you use -- OpenRouter's docs sample may not perfectly match
  what a given model/provider combination actually returns.
- Real-world behavior of `require_parameters` combined with `:free` model
  variants specifically (rate-limit/availability characteristics of free
  tiers can interact with routing in ways that are hard to predict from
  docs alone).

Recommended before trusting this across a full batch run: a single dialogue
or a small `--num-dialogues` run with `--target-llm-openrouter-require-parameters`
and `--reasoning-mode on`, checking that `reasoning_tokens_reported` and
`openrouter_served_by` show the expected provider on every turn, not just
the first.

## References

- OpenRouter, [Provider Selection](https://openrouter.ai/docs/guides/routing/provider-selection)
  -- the `provider` object fields (`order`, `allow_fallbacks`,
  `require_parameters`, and others not exposed as CLI flags here).
- OpenRouter, [Router Metadata](https://openrouter.ai/docs/guides/features/router-metadata)
  -- the `openrouter_metadata` response field, its experimental status, and
  the header(s) that enable it.
- [BerriAI/litellm#6857](https://github.com/BerriAI/litellm/issues/6857) --
  why a bare `provider=` keyword argument to `litellm.completion()` doesn't
  work reliably, and why `extra_body={"provider": {...}}` is used instead.
