"""The operation context manager, usage recording, token and request limits, time limits."""

from decimal import Decimal

import pytest
from tests.conftest import TEST_MODEL

from owl_agent_budget_guard import (
    BudgetExceededError,
    ExecutionTimeExceededError,
    InvalidValueError,
    LimitExceededError,
    Limits,
    Policies,
    UnknownCostError,
)


def _report_agent_budget(session) -> dict:
    return session.report()["agent"]["budget"]


def test_operation_charges_calculated_cost_on_success(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            op.record_llm_call(model=TEST_MODEL, input_tokens=1000, output_tokens=500)
        report = session.report()
    # 1000 * 1.00 / 1e6 + 500 * 5.00 / 1e6 = 0.0035
    assert report["agent"]["budget"]["actual_spending"] == "0.00350000"
    assert report["agent"]["budget"]["cost_by_source"]["calculated"] == "0.00350000"
    assert report["agent"]["budget"]["reserved_amount"] == "0.00000000"
    assert report["session"]["usage"]["input_tokens"] == 1000
    assert report["session"]["usage"]["output_tokens"] == 500
    assert report["session"]["usage"]["total_tokens"] == 1500
    assert report["session"]["usage"]["model_requests"] == 1
    assert report["session"]["usage"]["successful_operations"] == 1


def test_provider_reported_cost_takes_precedence_over_pricing(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            op.record_llm_call(
                model=TEST_MODEL, input_tokens=1000, output_tokens=500, reported_cost="0.02"
            )
        breakdown = session.report()["agent"]["budget"]["cost_by_source"]
    assert breakdown["reported"] == "0.02000000"
    assert breakdown["calculated"] == "0.00000000"


def test_failed_operation_charges_only_recorded_cost(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(RuntimeError, match="boom"):
            with session.operation("0.50", llm_calls=2) as op:
                op.record_llm_call(model=TEST_MODEL, input_tokens=100, output_tokens=0)
                raise RuntimeError("boom")
        report = session.report()
    budget = report["agent"]["budget"]
    assert budget["reserved_amount"] == "0.00000000"
    assert budget["actual_spending"] == "0.00010000"  # 100 * 1.00 / 1e6, the recorded call only
    assert report["session"]["usage"]["failed_operations"] == 1
    # Declared-but-unrecorded requests are counted on failure (conservative).
    assert report["session"]["usage"]["model_requests"] == 2


def test_failed_operation_with_nothing_recorded_is_not_charged(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(ValueError):
            with session.operation("0.50"):
                raise ValueError("failed before any model call")
        report = session.report()
    assert report["agent"]["budget"]["actual_spending"] == "0.00000000"
    assert report["agent"]["budget"]["remaining_available"] == "1.00000000"


def test_success_without_any_usage_charges_estimate_and_blocks(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(UnknownCostError, match="without recording any cost"):
            with session.operation("0.30"):
                pass
        report = session.report()
    assert report["agent"]["budget"]["actual_spending"] == "0.30000000"
    assert report["agent"]["budget"]["cost_by_source"]["estimated_fallback"] == "0.30000000"
    assert any(e["kind"] == "unreported_usage" for e in report["enforcement_events"])


def test_success_without_usage_under_estimate_policy_does_not_raise(
    make_guard, unique_name
) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1")),
        policies=Policies(unknown_cost="estimate"),
    )
    with guard.session(unique_name) as session:
        with session.operation("0.30"):
            pass
        assert session.report()["agent"]["budget"]["actual_spending"] == "0.30000000"


def test_explicit_zero_cost_is_accepted(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.30", tool_calls=1) as op:
            op.record_tool_call(cost=0)
        report = session.report()
    assert report["agent"]["budget"]["actual_spending"] == "0.00000000"
    assert report["session"]["usage"]["tool_executions"] == 1


def test_tool_call_without_cost_is_blocked(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(UnknownCostError, match="tool call cost was not provided"):
            with session.operation("0.30", tool_calls=1) as op:
                op.record_tool_call()
        assert session.report()["agent"]["budget"]["actual_spending"] == "0.30000000"


def test_unregistered_model_is_blocked_by_default(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(UnknownCostError, match="register pricing"):
            with session.operation("0.30", llm_calls=1) as op:
                op.record_llm_call(model="unpriced-model", input_tokens=10, output_tokens=10)


def test_unregistered_model_is_estimated_under_estimate_policy(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1")),
        policies=Policies(unknown_cost="estimate"),
    )
    with guard.session(unique_name) as session:
        with session.operation("0.30", llm_calls=1) as op:
            op.record_llm_call(model="unpriced-model", input_tokens=10, output_tokens=10)
        assert session.report()["agent"]["budget"]["actual_spending"] == "0.30000000"


def test_reserved_cost_is_held_while_operation_runs(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.40", tool_calls=1) as op:
            report = session.report()
            assert report["agent"]["budget"]["reserved_amount"] == "0.40000000"
            assert report["agent"]["budget"]["remaining_available"] == "0.60000000"
            op.record_tool_call(cost=0)


def test_actual_cost_above_estimate_fails_operation_and_is_recorded(
    make_guard, unique_name
) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(Exception) as info:
            with session.operation("0.001", llm_calls=1) as op:
                op.record_llm_call(model=TEST_MODEL, input_tokens=1_000_000, output_tokens=0)
        assert "exceeded" in str(info.value) or "reserved" in str(info.value)
        assert session.report()["agent"]["budget"]["actual_spending"] == "1.00000000"


def test_declared_llm_calls_cannot_be_exceeded(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(LimitExceededError, match="declared 1 model call"):
            with session.operation("0.10", llm_calls=1) as op:
                op.record_llm_call(reported_cost="0.01", input_tokens=1, output_tokens=1)
                op.record_llm_call(reported_cost="0.01", input_tokens=1, output_tokens=1)


def test_request_limit_is_enforced_before_execution(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1"), max_llm_calls=2),
        policies=Policies(on_budget_exceeded="raise"),
    )
    with guard.session(unique_name) as session:
        first = session.reserve("0.01", llm_calls=1)
        second = session.reserve("0.01", llm_calls=1)
        with pytest.raises(LimitExceededError, match="max_llm_calls=2"):
            session.reserve("0.01", llm_calls=1)
        first.record_llm_call(reported_cost="0.01")
        first.reconcile()
        second.record_llm_call(reported_cost="0.01")
        second.reconcile()
        with pytest.raises(LimitExceededError, match="model request limit reached"):
            session.check()
        with pytest.raises(LimitExceededError):
            session.reserve("0.01", llm_calls=1)


def test_unused_declared_requests_are_released_on_success(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1"), max_llm_calls=3))
    with guard.session(unique_name) as session:
        with session.operation("0.01", llm_calls=3) as op:
            op.record_llm_call(reported_cost="0.01", input_tokens=1, output_tokens=1)
        report = session.report()
    assert report["session"]["usage"]["model_requests"] == 1
    assert report["session"]["usage"]["model_requests_reserved"] == 0


def test_tool_call_limit_is_enforced(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1"), max_tool_calls=1),
        policies=Policies(on_budget_exceeded="raise"),
    )
    with guard.session(unique_name) as session:
        with pytest.raises(LimitExceededError, match="max_tool_calls"):
            session.reserve("0.01", tool_calls=2)
        with session.operation("0.01", tool_calls=1) as op:
            op.record_tool_call(cost=0)
        with pytest.raises(LimitExceededError, match="tool execution limit reached"):
            session.check()


def test_total_token_limit_is_enforced_after_recording(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1"), max_tokens=100),
        policies=Policies(on_budget_exceeded="raise"),
    )
    with guard.session(unique_name) as session:
        with pytest.raises(LimitExceededError, match="total token limit exceeded"):
            with session.operation("0.01", llm_calls=1) as op:
                op.record_llm_call(reported_cost="0.01", input_tokens=80, output_tokens=40)
        report = session.report()
        assert report["session"]["usage"]["total_tokens"] == 120
        assert report["session"]["budget"]["actual_spending"] == "0.01000000"
        with pytest.raises(LimitExceededError, match="token limit reached"):
            session.reserve("0.01")


def test_per_call_input_and_output_limits(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(
            max_cost=Decimal("1"), max_input_tokens_per_call=100, max_output_tokens_per_call=50
        ),
        policies=Policies(on_budget_exceeded="raise"),
    )
    with guard.session(unique_name) as session:
        with pytest.raises(LimitExceededError, match="input tokens for one call"):
            with session.operation("0.01", llm_calls=1) as op:
                op.record_llm_call(reported_cost="0.01", input_tokens=101, output_tokens=1)
    with guard.session("second") as session:
        with pytest.raises(LimitExceededError, match="output tokens for one call"):
            with session.operation("0.01", llm_calls=1) as op:
                op.record_llm_call(reported_cost="0.01", input_tokens=1, output_tokens=51)


def test_missing_token_usage_is_blocked_when_token_limits_exist(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1"), max_tokens=1000))
    with guard.session(unique_name) as session:
        with pytest.raises(UnknownCostError, match="token limits cannot be enforced"):
            with session.operation("0.10", llm_calls=1) as op:
                op.record_llm_call(reported_cost="0.01", input_tokens=None, output_tokens=5)


def test_missing_token_usage_without_token_limits_is_allowed_with_reported_cost(
    make_guard, unique_name
) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            op.record_llm_call(reported_cost="0.02")
        assert session.report()["agent"]["budget"]["actual_spending"] == "0.02000000"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_tokens": -1, "output_tokens": 0},
        {"input_tokens": 0, "output_tokens": -5},
        {"input_tokens": 1.5, "output_tokens": 0},
        {"input_tokens": True, "output_tokens": 0},
        {"input_tokens": 10, "output_tokens": 0, "cached_input_tokens": 11},
        {"input_tokens": 10, "output_tokens": 0, "cached_input_tokens": -1},
        {"input_tokens": 10, "output_tokens": 0, "model": "  "},
    ],
)
def test_invalid_usage_is_rejected_without_recording(make_guard, unique_name, kwargs) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            with pytest.raises(InvalidValueError):
                op.record_llm_call(reported_cost="0.01", **kwargs)
            op.record_llm_call(reported_cost="0.01", input_tokens=1, output_tokens=1)
        assert session.report()["session"]["usage"]["model_requests"] == 1


def test_cached_tokens_are_priced_and_reported(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            op.record_llm_call(
                model=TEST_MODEL, input_tokens=1000, output_tokens=0, cached_input_tokens=200
            )
        report = session.report()
    assert report["session"]["usage"]["cached_input_tokens"] == 200
    assert report["session"]["budget"]["actual_spending"] == "0.00082000"


def test_time_limit_blocks_new_operations(make_guard, unique_name, clock) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name, limits=Limits(max_execution_time=60)) as session:
        session.check()
        clock.advance(59)
        session.check()
        clock.advance(2)
        with pytest.raises(ExecutionTimeExceededError, match="max_execution_time"):
            session.check()
        with pytest.raises(ExecutionTimeExceededError):
            session.reserve("0.01")
        report = session.report()
    assert report["session"]["execution"]["remaining_seconds"] == 0.0


def test_time_limit_passed_during_operation_is_reported_after_settlement(
    make_guard, unique_name, clock
) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name, limits=Limits(max_execution_time=10)) as session:
        with pytest.raises(ExecutionTimeExceededError):
            with session.operation("0.10", llm_calls=1) as op:
                clock.advance(11)
                op.record_llm_call(reported_cost="0.05")
    assert session.report()["agent"]["budget"]["actual_spending"] == "0.00000000"


def test_time_limit_detected_at_settlement_still_charges_work(
    make_guard, unique_name, clock
) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name, limits=Limits(max_execution_time=10)) as session:
        with pytest.raises(ExecutionTimeExceededError):
            with session.operation("0.10", llm_calls=1) as op:
                op.record_llm_call(reported_cost="0.05")
                clock.advance(11)
        report = session.report()
    assert report["session"]["budget"]["actual_spending"] == "0.05000000"
    assert any(e["kind"] == "time_limit_exceeded" for e in report["enforcement_events"])


def test_budget_error_inside_operation_is_not_masked_by_settlement(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1")), policies=Policies(on_budget_exceeded="raise")
    )
    with guard.session(unique_name) as session:
        with pytest.raises(KeyError):
            with session.operation("0.10", tool_calls=1) as op:
                op.record_tool_call(cost=0)
                raise KeyError("agent error")


def test_reservation_settled_inside_block_is_not_settled_twice(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.50") as op:
            op.reconcile("0.20")
        assert op.state == "settled"
        assert session.report()["agent"]["budget"]["actual_spending"] == "0.20000000"


def test_operation_without_usage_is_charged_as_estimate_when_budget_policy_is_stop(
    make_guard, unique_name
) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("0.10")))
    with guard.session(unique_name) as session:
        with pytest.raises(UnknownCostError):
            with session.operation("0.10"):
                pass
        with pytest.raises(BudgetExceededError):
            session.reserve("0.01")
