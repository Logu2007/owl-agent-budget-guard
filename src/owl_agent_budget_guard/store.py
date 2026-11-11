"""SQLite persistence for budget state.

What is stored: agent and session counters, session states and limits, open
reservations, and the enforcement event log. Limits that come from code are
stored only for sessions, so a changed configuration applies to the agent on
the next start.

Concurrency model: one live guard owns a store file at a time. Opening a store
claims ownership with a random instance id. Every write checks that id inside
its transaction. If another process opens the same file, the first guard's next
write fails with :class:`StoreOwnershipError`, and that guard stops accepting
work. Budgets are never silently shared between two processes.

Crash recovery: a reservation that was still open when the process stopped is
settled on the next start as a failed operation, charged at the larger of its
known costs and its estimate. It is never assumed to be free.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from owl_agent_budget_guard.exceptions import PersistenceError, StoreOwnershipError
from owl_agent_budget_guard.ledger import COST_SOURCES, Scope
from owl_agent_budget_guard.limits import Limits

SCHEMA_VERSION = 1

_SCHEMA_STATEMENTS = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS scopes (name TEXT PRIMARY KEY, payload TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        state TEXT NOT NULL,
        scope_name TEXT NOT NULL,
        limits TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS reservations (
        reservation_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        payload TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        scope TEXT NOT NULL,
        kind TEXT NOT NULL,
        detail TEXT NOT NULL)""",
)


# ----------------------------------------------------------------------
# Serialization helpers (money is stored as exact decimal strings)
# ----------------------------------------------------------------------


def _money_or_none(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def limits_to_payload(limits: Limits) -> dict[str, Any]:
    return {
        "max_cost": _money_or_none(limits.max_cost),
        "max_operation_cost": _money_or_none(limits.max_operation_cost),
        "max_llm_calls": limits.max_llm_calls,
        "max_tool_calls": limits.max_tool_calls,
        "max_tokens": limits.max_tokens,
        "max_input_tokens_per_call": limits.max_input_tokens_per_call,
        "max_output_tokens_per_call": limits.max_output_tokens_per_call,
        "max_execution_time": limits.max_execution_time,
    }


def limits_from_payload(payload: dict[str, Any]) -> Limits:
    return Limits(
        max_cost=None if payload["max_cost"] is None else Decimal(payload["max_cost"]),
        max_operation_cost=(
            None
            if payload["max_operation_cost"] is None
            else Decimal(payload["max_operation_cost"])
        ),
        max_llm_calls=payload["max_llm_calls"],
        max_tool_calls=payload["max_tool_calls"],
        max_tokens=payload["max_tokens"],
        max_input_tokens_per_call=payload["max_input_tokens_per_call"],
        max_output_tokens_per_call=payload["max_output_tokens_per_call"],
        max_execution_time=payload["max_execution_time"],
    )


def scope_to_payload(scope: Scope, now: float) -> dict[str, Any]:
    return {
        "actual_cost": str(scope.actual_cost),
        "reserved_cost": str(scope.reserved_cost),
        "cost_by_source": {source: str(scope.cost_by_source[source]) for source in COST_SOURCES},
        "llm_calls_used": scope.llm_calls_used,
        "llm_calls_reserved": scope.llm_calls_reserved,
        "tool_calls_used": scope.tool_calls_used,
        "tool_calls_reserved": scope.tool_calls_reserved,
        "input_tokens": scope.input_tokens,
        "output_tokens": scope.output_tokens,
        "cached_input_tokens": scope.cached_input_tokens,
        "successful_operations": scope.successful_operations,
        "failed_operations": scope.failed_operations,
        "operations_seconds_total": scope.operations_seconds_total,
        # Elapsed time is stored instead of a monotonic timestamp, which does not
        # survive a restart. On load, the scope resumes with this much time used.
        "elapsed_seconds": scope.elapsed(now),
    }


def scope_from_payload(name: str, limits: Limits, payload: dict[str, Any], now: float) -> Scope:
    costs = payload["cost_by_source"]
    return Scope(
        name=name,
        limits=limits,
        started_at=now - float(payload["elapsed_seconds"]),
        actual_cost=Decimal(payload["actual_cost"]),
        reserved_cost=Decimal(payload["reserved_cost"]),
        cost_by_source={source: Decimal(costs.get(source, "0")) for source in COST_SOURCES},
        llm_calls_used=payload["llm_calls_used"],
        llm_calls_reserved=payload["llm_calls_reserved"],
        tool_calls_used=payload["tool_calls_used"],
        tool_calls_reserved=payload["tool_calls_reserved"],
        input_tokens=payload["input_tokens"],
        output_tokens=payload["output_tokens"],
        cached_input_tokens=payload["cached_input_tokens"],
        successful_operations=payload["successful_operations"],
        failed_operations=payload["failed_operations"],
        operations_seconds_total=float(payload["operations_seconds_total"]),
    )


@dataclass(frozen=True)
class SessionRow:
    """A session as stored: identity, state, scope name and limits."""

    session_id: str
    name: str
    state: str
    scope_name: str
    limits: dict[str, Any]


@dataclass
class StoreSnapshot:
    """Everything needed to restore a guard, read in one consistent transaction."""

    owner: str | None
    agent_name: str | None
    currency: str | None
    scopes: dict[str, dict[str, Any]] = field(default_factory=dict)
    sessions: dict[str, SessionRow] = field(default_factory=dict)
    reservations: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    events: list[dict[str, str]] = field(default_factory=list)


# ----------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------


class SqliteStore:
    """A budget state file backed by SQLite.

    Open with ``SqliteStore(path)`` for a live guard. Open with
    ``SqliteStore(path, read_only=True)`` to inspect a store from the CLI without
    claiming ownership.
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._io_lock = threading.Lock()
        try:
            if read_only:
                if not self.path.exists():
                    raise PersistenceError(f"no budget store at {self.path}")
                uri = f"{self.path.resolve().as_uri()}?mode=ro"
                self._conn = sqlite3.connect(
                    uri, uri=True, isolation_level=None, check_same_thread=False
                )
            else:
                self._conn = sqlite3.connect(
                    str(self.path), isolation_level=None, check_same_thread=False, timeout=10
                )
                self._conn.execute("PRAGMA journal_mode=WAL")
                try:
                    self._migrate()
                except BaseException:
                    self._conn.close()
                    raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"cannot open budget store {self.path}: {exc}") from exc

    def close(self) -> None:
        with self._io_lock:
            self._conn.close()

    # -- transactions -------------------------------------------------

    def _migrate(self) -> None:
        with self._write_transaction() as conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif row[0] != str(SCHEMA_VERSION):
                raise PersistenceError(
                    f"budget store schema version {row[0]} is not supported "
                    f"(this release reads version {SCHEMA_VERSION})"
                )

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one write transaction. Rolls back on any error."""
        with self._io_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    yield self._conn
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                raise PersistenceError(f"budget store write failed: {exc}") from exc

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._io_lock:
            try:
                self._conn.execute("BEGIN")
                try:
                    yield self._conn
                finally:
                    self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                raise PersistenceError(f"budget store read failed: {exc}") from exc

    # -- ownership ----------------------------------------------------

    def claim(self, agent_name: str, currency: str, owner: str) -> None:
        """Bind this store to one agent and currency, and take ownership for ``owner``.

        Raises:
            PersistenceError: if the store belongs to a different agent or currency.
        """
        with self._write_transaction() as conn:
            found = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            if "agent_name" in found and (
                found["agent_name"] != agent_name or found["currency"] != currency
            ):
                raise PersistenceError(
                    f"budget store {self.path} belongs to agent {found['agent_name']!r} in "
                    f"{found['currency']}; it cannot be used for {agent_name!r} in {currency}"
                )
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [("agent_name", agent_name), ("currency", currency), ("owner", owner)],
            )

    # -- reads --------------------------------------------------------

    def load(self) -> StoreSnapshot:
        """Return the full stored state."""
        with self._read_transaction() as conn:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            snapshot = StoreSnapshot(
                owner=meta.get("owner"),
                agent_name=meta.get("agent_name"),
                currency=meta.get("currency"),
            )
            for name, payload in conn.execute("SELECT name, payload FROM scopes"):
                snapshot.scopes[name] = json.loads(payload)
            for row in conn.execute(
                "SELECT session_id, name, state, scope_name, limits FROM sessions"
            ):
                snapshot.sessions[row[0]] = SessionRow(
                    session_id=row[0],
                    name=row[1],
                    state=row[2],
                    scope_name=row[3],
                    limits=json.loads(row[4]),
                )
            for rid, payload in conn.execute("SELECT reservation_id, payload FROM reservations"):
                snapshot.reservations.append((rid, json.loads(payload)))
            for timestamp, scope, kind, detail in conn.execute(
                "SELECT timestamp, scope, kind, detail FROM events ORDER BY id"
            ):
                snapshot.events.append(
                    {"timestamp": timestamp, "scope": scope, "kind": kind, "detail": detail}
                )
            return snapshot

    # -- writes -------------------------------------------------------

    def save(
        self,
        *,
        owner: str,
        scopes: dict[str, dict[str, Any]],
        sessions: list[SessionRow],
        open_reservations: dict[str, list[tuple[str, dict[str, Any]]]],
        events: list[dict[str, str]],
    ) -> None:
        """Write one consistent update. All of it commits, or none of it does.

        Args:
            owner: The instance id that claimed the store. Must still match.
            scopes: Scope name to payload.
            sessions: Session rows to upsert.
            open_reservations: Session id to its complete list of open reservations.
                Rows for that session are replaced, so settled ones disappear.
            events: Enforcement events to append.
        """
        with self._write_transaction() as conn:
            self._check_owner(conn, owner)
            for name, payload in scopes.items():
                conn.execute(
                    "INSERT INTO scopes(name, payload) VALUES(?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
                    (name, json.dumps(payload)),
                )
            for row in sessions:
                conn.execute(
                    "INSERT INTO sessions(session_id, name, state, scope_name, limits) "
                    "VALUES(?, ?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                    "name=excluded.name, state=excluded.state, limits=excluded.limits",
                    (row.session_id, row.name, row.state, row.scope_name, json.dumps(row.limits)),
                )
            for session_id, items in open_reservations.items():
                conn.execute("DELETE FROM reservations WHERE session_id=?", (session_id,))
                conn.executemany(
                    "INSERT INTO reservations(reservation_id, session_id, payload) VALUES(?, ?, ?)",
                    [(rid, session_id, json.dumps(payload)) for rid, payload in items],
                )
            conn.executemany(
                "INSERT INTO events(timestamp, scope, kind, detail) VALUES(?, ?, ?, ?)",
                [(e["timestamp"], e["scope"], e["kind"], e["detail"]) for e in events],
            )

    def _check_owner(self, conn: sqlite3.Connection, owner: str) -> None:
        row = conn.execute("SELECT value FROM meta WHERE key='owner'").fetchone()
        if row is None or row[0] != owner:
            raise StoreOwnershipError(
                f"budget store {self.path} is now owned by another guard instance; "
                "this guard can no longer write to it and must be reopened"
            )
