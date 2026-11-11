"""OpenAI integration, tested with the real openai response types (no network)."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest
from tests.conftest import TEST_MODEL

from owl_agent_budget_guard import Limits, Policies, UnknownCostError
from owl_agent_budget_guard.integrations.openai_sdk import (
    acreate_and_record,
    create_and_record,
    extract_usage,
    record_openai_usage,
)

openai_types = pytest.importorskip("openai.types")
from openai.types import CompletionUsage  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402
from openai.types.completion_usage import PromptTokensDetails  # noqa: E402
from openai.types.responses import Response  # noqa: E402
from openai.types.responses.response_usage import (  # noqa: E402
    InputTokensDetails,
    ResponseUsage,
)


def _chat_completion(prompt: int, completion: int, cached: int = 0) -> ChatCompletion:
    usage = CompletionUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
    )
    return ChatCompletion.model_construct(
        id="chatcmpl-test",
        choices=[],
        created=0,
        model=TEST_MODEL,
        object="chat.completion",
        usage=usage,
    )


def test_chat_completion_usage_is_extracted_and_priced(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            record_openai_usage(op, _chat_completion(prompt=1000, completion=500, cached=200))
        report = session.report()
    # 800 uncached * 1.00 + 200 cached * 0.10 + 500 output * 5.00, per million
    assert report["session"]["budget"]["actual_spending"] == "0.00332000"
    assert report["session"]["usage"]["input_tokens"] == 1000
    assert report["session"]["usage"]["output_tokens"] == 500
    assert report["session"]["usage"]["cached_input_tokens"] == 200


def test_responses_api_usage_is_extracted(make_guard, unique_name) -> None:
    usage = ResponseUsage(
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
        output_tokens_details={"reasoning_tokens": 0},
    )
    response = Response.model_construct(model=TEST_MODEL, usage=usage)
    assert extract_usage(response) == (1000, 500, 0, TEST_MODEL)

    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            record_openai_usage(op, response)
        assert session.report()["session"]["budget"]["actual_spending"] == "0.00350000"


def test_dict_shaped_responses_are_supported() -> None:
    response = {
        "model": TEST_MODEL,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }
    assert extract_usage(response) == (10, 2, 4, TEST_MODEL)


def test_missing_or_non_integer_usage_is_reported_as_unknown() -> None:
    assert extract_usage(SimpleNamespace(model="m", usage=None)) == (None, None, None, "m")
    odd = SimpleNamespace(
        model=None, usage=SimpleNamespace(prompt_tokens="10", completion_tokens=True)
    )
    assert extract_usage(odd) == (None, None, None, None)


def test_missing_usage_is_blocked_by_default(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    response = SimpleNamespace(model=TEST_MODEL, usage=None)
    with guard.session(unique_name) as session:
        with pytest.raises(UnknownCostError):
            with session.operation("0.10", llm_calls=1) as op:
                record_openai_usage(op, response)
        assert session.report()["session"]["budget"]["actual_spending"] == "0.10000000"


def test_missing_usage_is_estimated_under_estimate_policy(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1")), policies=Policies(unknown_cost="estimate")
    )
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            record_openai_usage(op, SimpleNamespace(model=TEST_MODEL, usage=None))
        assert session.report()["session"]["budget"]["actual_spending"] == "0.10000000"


def test_create_and_record_uses_requested_model_and_returns_response(
    make_guard, unique_name
) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return _chat_completion(prompt=1000, completion=0)

    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            response = create_and_record(op, fake_create, model=TEST_MODEL, messages=[])
        assert response.model == TEST_MODEL
        assert calls == [{"model": TEST_MODEL, "messages": []}]  # every kwarg reaches create
        assert session.report()["session"]["budget"]["actual_spending"] == "0.00100000"


def test_failed_api_call_records_nothing(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))

    def failing_create(**kwargs):
        raise ConnectionError("network down")

    with guard.session(unique_name) as session:
        with pytest.raises(ConnectionError):
            with session.operation("0.10", llm_calls=1) as op:
                create_and_record(op, failing_create, model=TEST_MODEL)
        report = session.report()
    assert report["session"]["budget"]["actual_spending"] == "0.00000000"
    assert report["session"]["usage"]["failed_operations"] == 1


def test_async_create_and_record(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))

    async def fake_async_create(**kwargs):
        return _chat_completion(prompt=0, completion=1000)

    async def run(session):
        with session.operation("0.10", llm_calls=1) as op:
            return await acreate_and_record(op, fake_async_create, model=TEST_MODEL)

    with guard.session(unique_name) as session:
        asyncio.run(run(session))
        assert session.report()["session"]["budget"]["actual_spending"] == "0.00500000"


def test_token_limits_apply_to_openai_usage(make_guard, unique_name) -> None:
    from owl_agent_budget_guard import LimitExceededError

    guard = make_guard(
        limits=Limits(max_cost=Decimal("1"), max_output_tokens_per_call=100),
        policies=Policies(on_budget_exceeded="raise"),
    )
    with guard.session(unique_name) as session:
        with pytest.raises(LimitExceededError, match="output tokens for one call"):
            with session.operation("0.10", llm_calls=1) as op:
                record_openai_usage(op, _chat_completion(prompt=10, completion=101))
