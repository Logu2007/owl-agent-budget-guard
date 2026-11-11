"""Usage reports as plain dictionaries, plus a human-readable text renderer.

The dictionary is JSON-compatible. Money values are strings so that no precision
is lost. ``None`` means "not limited" or "not applicable".
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from owl_agent_budget_guard.ledger import COST_SOURCES, Scope
from owl_agent_budget_guard.money import MONEY_QUANTUM, ZERO

REPORT_VERSION = 1

NOTES: tuple[str, ...] = (
    "reserved_amount is an estimate held for in-flight operations. It is not spent.",
    "actual_spending is the sum of settled amounts. Its cost_by_source shows how each "
    "amount was determined.",
    "estimated_fallback is a conservative charge used when no cost was known. It is not "
    "a provider billing amount.",
    "calculated costs come from locally registered prices. They are not provider invoices.",
)


def _money(value: Decimal | None) -> str | None:
    """Render a money value with exactly eight decimal places."""
    return None if value is None else format(value.quantize(MONEY_QUANTUM), "f")


def _scope_block(scope: Scope, now: float) -> dict[str, Any]:
    limits = scope.limits
    remaining = scope.remaining_cost()
    remaining_time = scope.remaining_execution_time(now)
    return {
        "budget": {
            "limit": _money(limits.max_cost),
            "max_operation_cost": _money(limits.max_operation_cost),
            "actual_spending": _money(scope.actual_cost),
            "reserved_amount": _money(scope.reserved_cost),
            "remaining_available": _money(remaining),
            "cost_by_source": {
                source: _money(scope.cost_by_source[source]) for source in COST_SOURCES
            },
        },
        "usage": {
            "input_tokens": scope.input_tokens,
            "output_tokens": scope.output_tokens,
            "total_tokens": scope.total_tokens,
            "cached_input_tokens": scope.cached_input_tokens,
            "token_limit": limits.max_tokens,
            "model_requests": scope.llm_calls_used,
            "model_requests_reserved": scope.llm_calls_reserved,
            "model_request_limit": limits.max_llm_calls,
            "tool_executions": scope.tool_calls_used,
            "tool_executions_reserved": scope.tool_calls_reserved,
            "tool_execution_limit": limits.max_tool_calls,
            "successful_operations": scope.successful_operations,
            "failed_operations": scope.failed_operations,
        },
        "execution": {
            "elapsed_seconds": round(scope.elapsed(now), 3),
            "time_limit_seconds": limits.max_execution_time,
            "remaining_seconds": None if remaining_time is None else round(remaining_time, 3),
            "operations_seconds_total": round(scope.operations_seconds_total, 3),
        },
        "budget_exhausted": remaining is not None and remaining <= ZERO,
    }


def build_session_report(
    *,
    agent_name: str,
    currency: str,
    session_id: str,
    session_name: str,
    session_state: str,
    agent_scope: Scope,
    session_scope: Scope,
    agent_now: float,
    session_now: float,
    events: list[dict[str, str]],
) -> dict[str, Any]:
    """Assemble the report for one session, including the agent-level totals."""
    relevant_scopes = {agent_scope.name, session_scope.name}
    return {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "currency": currency,
        "agent": {"name": agent_name, **_scope_block(agent_scope, agent_now)},
        "session": {
            "id": session_id,
            "name": session_name,
            "state": session_state,
            **_scope_block(session_scope, session_now),
        },
        "enforcement_events": [event for event in events if event["scope"] in relevant_scopes],
        "notes": list(NOTES),
    }


def _fmt_seconds(value: float | None) -> str:
    return "unlimited" if value is None else f"{value} s"


def _fmt_limit(value: Any, currency: str | None = None) -> str:
    if value is None:
        return "unlimited"
    if currency is not None:
        return f"{value} {currency}"
    return str(value)


def _budget_lines(title: str, block: dict[str, Any], currency: str) -> list[str]:
    budget = block["budget"]
    lines = [
        title,
        f"  Limit:                 {_fmt_limit(budget['limit'], currency)}",
        f"  Actual spending:       {budget['actual_spending']} {currency}",
        f"  Reserved (estimate):   {budget['reserved_amount']} {currency}",
        f"  Remaining available:   {_fmt_limit(budget['remaining_available'], currency)}",
        "  Cost by source:",
    ]
    for source, amount in budget["cost_by_source"].items():
        lines.append(f"    {source:<18} {amount} {currency}")
    return lines


def _usage_lines(title: str, block: dict[str, Any]) -> list[str]:
    usage = block["usage"]
    execution = block["execution"]
    return [
        title,
        f"  Input tokens:          {usage['input_tokens']}",
        f"  Output tokens:         {usage['output_tokens']}",
        f"  Total tokens:          {usage['total_tokens']} "
        f"(limit {_fmt_limit(usage['token_limit'])})",
        f"  Model requests:        {usage['model_requests']} "
        f"(limit {_fmt_limit(usage['model_request_limit'])})",
        f"  Tool executions:       {usage['tool_executions']} "
        f"(limit {_fmt_limit(usage['tool_execution_limit'])})",
        f"  Operations:            {usage['successful_operations']} succeeded, "
        f"{usage['failed_operations']} failed",
        f"  Elapsed time:          {execution['elapsed_seconds']} s "
        f"(limit {_fmt_seconds(execution['time_limit_seconds'])})",
    ]


def format_report_text(report: dict[str, Any]) -> str:
    """Render a report dictionary as readable plain text."""
    currency = report["currency"]
    agent = report["agent"]
    session = report["session"]

    lines = [
        "OWL Agent Budget Guard report",
        f"Generated at: {report['generated_at']}",
        "",
        f"Agent:   {agent['name']}",
        f"Session: {session['name']} ({session['id']})",
        f"State:   {session['state']}"
        + (
            "  [budget exhausted]"
            if session["budget_exhausted"] or agent["budget_exhausted"]
            else ""
        ),
        "",
        "Reserved amounts are estimates, not spending. Cost by source shows how each "
        "actual amount was determined.",
        "",
    ]
    lines += _budget_lines("Agent budget (all sessions)", agent, currency)
    lines.append("")
    lines += _budget_lines("Session budget", session, currency)
    lines.append("")
    lines += _usage_lines("Session usage", session)
    lines.append("")
    lines += ["Enforcement events"]
    events = report["enforcement_events"]
    if not events:
        lines.append("  none")
    for event in events:
        lines.append(
            f"  [{event['timestamp']}] {event['kind']} ({event['scope']}): {event['detail']}"
        )
    lines += ["", "Notes"]
    lines += [f"  - {note}" for note in report["notes"]]
    if report.get("note"):
        lines += ["", report["note"]]
    return "\n".join(lines)
