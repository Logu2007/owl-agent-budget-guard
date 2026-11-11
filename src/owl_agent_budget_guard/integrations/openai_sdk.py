"""OpenAI SDK integration: record usage from responses into a budget operation.

This module reads usage fields by name from the response object. It does not
import the ``openai`` package, so the core library has no dependency on it.
Install the SDK separately (``pip install openai``) to make calls.

Supported responses:

* Chat Completions (``client.chat.completions.create``): ``usage.prompt_tokens``,
  ``usage.completion_tokens``, ``usage.prompt_tokens_details.cached_tokens``.
* Responses API (``client.responses.create``): ``usage.input_tokens``,
  ``usage.output_tokens``, ``usage.input_tokens_details.cached_tokens``.

Limitations:

* Cache-write tokens (reported by some providers) are billed at the normal input
  rate here. Cost is only as accurate as the prices you register.
* OpenAI does not return a cost in the response, so cost comes from registered
  pricing under the model name you pass, or from ``reported_cost``. If the
  response's model name differs from the registered name, pass ``model=``
  explicitly.
* Output tokens are checked after the call. To cap output before the call, set
  ``max_completion_tokens`` (Chat Completions) or ``max_output_tokens``
  (Responses) from ``max_output_tokens_per_call``.

Example::

    from owl_agent_budget_guard.integrations.openai_sdk import create_and_record

    with session.operation("0.02", llm_calls=1) as op:
        response = create_and_record(
            op, client.chat.completions.create,
            model="gpt-4o-mini", messages=messages,
        )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from owl_agent_budget_guard.reservation import Reservation


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _count(value: Any) -> int | None:
    # Only real integers count. Anything else is treated as "not reported".
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def extract_usage(response: Any) -> tuple[int | None, int | None, int | None, str | None]:
    """Return ``(input_tokens, output_tokens, cached_input_tokens, model)`` from a response.

    Missing values are ``None``. A response with no usage yields all ``None``
    token counts, which the budget then handles under its ``unknown_cost`` policy.
    """
    model = _field(response, "model")
    usage = _field(response, "usage")
    if usage is None:
        return None, None, None, model if isinstance(model, str) else None

    if _field(usage, "prompt_tokens") is not None or _field(usage, "completion_tokens") is not None:
        input_tokens = _count(_field(usage, "prompt_tokens"))
        output_tokens = _count(_field(usage, "completion_tokens"))
        details = _field(usage, "prompt_tokens_details")
    else:
        input_tokens = _count(_field(usage, "input_tokens"))
        output_tokens = _count(_field(usage, "output_tokens"))
        details = _field(usage, "input_tokens_details")
    cached = _count(_field(details, "cached_tokens"))
    return input_tokens, output_tokens, cached, model if isinstance(model, str) else None


def record_openai_usage(
    reservation: Reservation,
    response: Any,
    *,
    model: str | None = None,
) -> None:
    """Record one completed OpenAI call into ``reservation``.

    Args:
        reservation: The operation the call belongs to.
        response: A Chat Completions or Responses API result.
        model: Registered model name for pricing. Defaults to the response's model.
    """
    input_tokens, output_tokens, cached, response_model = extract_usage(response)
    reservation.record_llm_call(
        model=model if model is not None else response_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached or 0,
    )


def create_and_record(
    reservation: Reservation,
    create: Callable[..., Any],
    *args: Any,
    pricing_model: str | None = None,
    **kwargs: Any,
) -> Any:
    """Call an OpenAI create method, then record its usage. Returns the response.

    All keyword arguments go to ``create`` unchanged. ``pricing_model`` is the one
    exception: it names the registered price to use when that differs from the
    request's ``model``. If the call raises, nothing is recorded and the error
    propagates, so the operation settles as a failure.
    """
    response = create(*args, **kwargs)
    record_openai_usage(
        reservation,
        response,
        model=pricing_model if pricing_model is not None else kwargs.get("model"),
    )
    return response


async def acreate_and_record(
    reservation: Reservation,
    create: Callable[..., Awaitable[Any]],
    *args: Any,
    pricing_model: str | None = None,
    **kwargs: Any,
) -> Any:
    """Async version of :func:`create_and_record` for ``AsyncOpenAI`` clients."""
    response = await create(*args, **kwargs)
    record_openai_usage(
        reservation,
        response,
        model=pricing_model if pricing_model is not None else kwargs.get("model"),
    )
    return response
