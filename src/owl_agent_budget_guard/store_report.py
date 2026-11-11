"""Read-only views of a budget store, for the CLI.

Opening the store read-only never claims ownership, so inspecting a budget
cannot disable the guard that is running against it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from owl_agent_budget_guard.exceptions import PersistenceError
from owl_agent_budget_guard.ledger import Scope
from owl_agent_budget_guard.limits import Limits
from owl_agent_budget_guard.money import MONEY_QUANTUM
from owl_agent_budget_guard.report import (
    NOTES,
    REPORT_VERSION,
    _budget_lines,
    _scope_block,
)
from owl_agent_budget_guard.store import SqliteStore, scope_from_payload


def _money(value: str | None) -> str:
    return format(Decimal(value or "0").quantize(MONEY_QUANTUM), "f")


def build_store_report(
    path: str | Path,
    *,
    agent_name: str,
    currency: str,
    limits: Limits,
) -> dict[str, Any]:
    """Build a report from a stored budget. Raises :class:`PersistenceError`."""
    store = SqliteStore(path, read_only=True)
    try:
        snapshot = store.load()
    finally:
        store.close()

    if snapshot.agent_name is None:
        raise PersistenceError(f"budget store {path} has no recorded agent yet")
    if snapshot.agent_name != agent_name or snapshot.currency != currency:
        raise PersistenceError(
            f"budget store {path} belongs to agent {snapshot.agent_name!r} in "
            f"{snapshot.currency}, not {agent_name!r} in {currency}"
        )

    now = time.monotonic()
    agent_key = f"agent:{agent_name}"
    payload = snapshot.scopes.get(agent_key)
    agent_scope = (
        scope_from_payload(agent_key, limits, payload, now)
        if payload is not None
        else Scope(name=agent_key, limits=limits, started_at=now)
    )

    sessions = []
    for row in snapshot.sessions.values():
        session_payload = snapshot.scopes.get(row.scope_name, {})
        sessions.append(
            {
                "id": row.session_id,
                "name": row.name,
                "state": row.state,
                "actual_spending": _money(session_payload.get("actual_cost")),
                "reserved_amount": _money(session_payload.get("reserved_cost")),
            }
        )

    return {
        "report_version": REPORT_VERSION,
        "source": "store",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "store_path": str(path),
        "currency": currency,
        "agent": {"name": agent_name, **_scope_block(agent_scope, now)},
        "sessions": sessions,
        "enforcement_events": snapshot.events,
        "notes": list(NOTES),
    }


def format_store_report_text(report: dict[str, Any]) -> str:
    """Render a stored-budget report as plain text."""
    currency = report["currency"]
    agent = report["agent"]
    lines = [
        "OWL Agent Budget Guard: stored budget",
        f"Store:        {report['store_path']}",
        f"Generated at: {report['generated_at']}",
        "",
        f"Agent:        {agent['name']}",
        "",
    ]
    lines += _budget_lines("Agent budget (all sessions)", agent, currency)
    lines += ["", f"Sessions ({len(report['sessions'])})"]
    if not report["sessions"]:
        lines.append("  none")
    for session in report["sessions"]:
        lines.append(
            f"  {session['id']}  {session['state']:<8}  {session['name']}: "
            f"spent {session['actual_spending']}, reserved {session['reserved_amount']} {currency}"
        )
    lines += ["", f"Enforcement events ({len(report['enforcement_events'])})"]
    for event in report["enforcement_events"][-20:]:
        lines.append(
            f"  [{event['timestamp']}] {event['kind']} ({event['scope']}): {event['detail']}"
        )
    lines += ["", "Notes"] + [f"  - {note}" for note in report["notes"]]
    return "\n".join(lines)
