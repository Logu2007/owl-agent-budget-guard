"""BudgetGuard: the entry point that owns agent-wide budget state."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from owl_agent_budget_guard.exceptions import PersistenceError, PricingError
from owl_agent_budget_guard.ledger import Scope
from owl_agent_budget_guard.limits import Limits, Policies
from owl_agent_budget_guard.money import ZERO
from owl_agent_budget_guard.pricing import ModelPricing, PricingRegistry
from owl_agent_budget_guard.reservation import apply_settlement
from owl_agent_budget_guard.session import Session
from owl_agent_budget_guard.store import (
    SessionRow,
    SqliteStore,
    StoreSnapshot,
    limits_from_payload,
    scope_from_payload,
    scope_to_payload,
)
from owl_agent_budget_guard.validation import validate_currency, validate_name


class BudgetGuard:
    """Owns the agent-wide budget and creates sessions.

    The agent-wide limits apply to the sum of all sessions created from this
    guard. Each session may add its own, stricter limits. An operation must fit
    within every applicable scope.

    Persistence: pass ``store=SqliteStore(path)`` to keep the budget across
    restarts. Without a store, all state lives in memory.

    Concurrency: all accounting for one guard is protected by one re-entrant
    lock. Threads in one process are safe. Two live guards on the same store
    file are not: the newer one takes ownership, and the older one fails its next
    write.

    Example::

        guard = BudgetGuard(
            agent_name="website-analyser",
            currency="USD",
            limits=Limits(max_cost=Decimal("5"), max_llm_calls=50),
            store=SqliteStore("owl-budget.db"),
        )
        with guard.session("task-1", limits=Limits(max_cost=Decimal("1"))) as session:
            with session.operation(estimated_cost="0.10", llm_calls=1) as op:
                ...
    """

    def __init__(
        self,
        *,
        agent_name: str = "default-agent",
        currency: str = "USD",
        limits: Limits | None = None,
        policies: Policies | None = None,
        pricing: Iterable[ModelPricing] = (),
        store: SqliteStore | None = None,
    ) -> None:
        self._agent_name = validate_name(agent_name, "agent_name")
        self._currency = validate_currency(currency)
        if limits is not None and not isinstance(limits, Limits):
            raise TypeError("limits must be a Limits instance")
        if policies is not None and not isinstance(policies, Policies):
            raise TypeError("policies must be a Policies instance")
        if store is not None and not isinstance(store, SqliteStore):
            raise TypeError("store must be a SqliteStore instance")
        self._limits = limits if limits is not None else Limits()
        self._policies = policies if policies is not None else Policies()
        self._lock = threading.RLock()
        self._events: list[dict[str, str]] = []
        self._pending_events: list[dict[str, str]] = []
        self._pricing = PricingRegistry()
        self._agent_scope = Scope(
            name=f"agent:{self._agent_name}",
            limits=self._limits,
            started_at=time.monotonic(),
        )
        self._store = store
        self._instance_id = uuid.uuid4().hex
        self._faulted = False
        self._live_session_ids: set[str] = set()
        self._restored_sessions: dict[str, SessionRow] = {}
        self._restored_payloads: dict[str, dict[str, Any]] = {}
        for item in pricing:
            self.register_pricing(item)
        if store is not None:
            try:
                self._open_store(store)
            except BaseException:
                store.close()  # do not leak the connection when the guard cannot start
                raise

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def limits(self) -> Limits:
        return self._limits

    @property
    def policies(self) -> Policies:
        return self._policies

    @property
    def pricing(self) -> PricingRegistry:
        return self._pricing

    @property
    def store(self) -> SqliteStore | None:
        return self._store

    def close(self) -> None:
        """Close the store connection. Open sessions should be closed first."""
        with self._lock:
            if self._store is not None:
                self._store.close()

    def __enter__(self) -> BudgetGuard:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False

    def register_pricing(self, pricing: ModelPricing, *, replace: bool = False) -> None:
        """Register prices for a model. The currency must match the guard currency."""
        if not isinstance(pricing, ModelPricing):
            raise PricingError("register_pricing() expects a ModelPricing instance")
        if pricing.currency != self._currency:
            raise PricingError(
                f"pricing for model {pricing.model!r} is in {pricing.currency}, "
                f"but this guard uses {self._currency}"
            )
        with self._lock:
            self._pricing.register(pricing, replace=replace)

    def session(
        self,
        name: str,
        *,
        limits: Limits | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create a new session, or resume a stored one.

        Args:
            name: Human-readable label, for example ``"website-analysis"``.
            limits: Session-level limits. When resuming, these replace the stored limits.
            session_id: Resume the session with this id from the store. The guard
                must have a store, and ``name`` must match the stored name.
        """
        if session_id is not None:
            return self._resume_session(name, limits, session_id)
        return Session(
            guard=self,
            name=name,
            limits=limits if limits is not None else Limits(),
        )

    def enforcement_events(self) -> list[dict[str, str]]:
        """Return a copy of all enforcement events, including those loaded from the store."""
        with self._lock:
            return [dict(event) for event in self._events]

    # ------------------------------------------------------------------
    # Internal: events, mutation, persistence
    # ------------------------------------------------------------------

    def _record_event(self, *, scope: str, kind: str, detail: str) -> None:
        # Caller must hold the lock.
        event = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "scope": scope,
            "kind": kind,
            "detail": detail,
        }
        self._events.append(event)
        self._pending_events.append(event)

    @contextmanager
    def _mutation(self, session: Session | None) -> Iterator[None]:
        """Hold the lock for one state change, then persist it whatever the outcome.

        Persistence runs even when the block raises, because a rejection can still
        change session state and record events.
        """
        with self._lock:
            if self._faulted:
                raise PersistenceError(
                    "the budget store is unavailable after a failed write; "
                    "reopen the guard to continue"
                )
            try:
                yield
            finally:
                self._flush(session)

    def _flush(self, session: Session | None) -> None:
        # Caller must hold the lock.
        if self._store is None:
            self._pending_events.clear()
            return
        now = time.monotonic()
        scopes: dict[str, dict[str, Any]] = {
            self._agent_scope.name: scope_to_payload(self._agent_scope, now)
        }
        sessions: list[SessionRow] = []
        open_reservations: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        if session is not None:
            scopes[session._scope.name] = scope_to_payload(session._scope, now)
            sessions.append(session._row())
            open_reservations[session.session_id] = [
                (rid, reservation.to_payload(now))
                for rid, reservation in session._open_reservations.items()
            ]
        try:
            self._store.save(
                owner=self._instance_id,
                scopes=scopes,
                sessions=sessions,
                open_reservations=open_reservations,
                events=list(self._pending_events),
            )
        except BaseException:
            # In-memory state may now be ahead of the store. Refuse further work.
            self._faulted = True
            raise
        self._pending_events.clear()

    def _attach_session(self, session: Session) -> None:
        """Register a session with this guard and persist it (new or resumed)."""
        with self._mutation(session):
            if session.session_id in self._live_session_ids:
                raise PersistenceError(
                    f"session {session.session_id} is already attached to this guard"
                )
            self._live_session_ids.add(session.session_id)

    def _resume_session(self, name: str, limits: Limits | None, session_id: str) -> Session:
        if self._store is None:
            raise PersistenceError("resuming a session requires a guard with a store")
        with self._lock:
            if session_id in self._live_session_ids:
                raise PersistenceError(f"session {session_id} is already attached to this guard")
            row = self._restored_sessions.get(session_id)
            if row is None:
                raise PersistenceError(f"session {session_id} is not in the budget store")
            if row.name != name:
                raise PersistenceError(f"session {session_id} is named {row.name!r}, not {name!r}")
            payload = self._restored_payloads.get(row.scope_name)
            if payload is None:
                raise PersistenceError(f"session {session_id} has no stored counters")
            effective = limits if limits is not None else limits_from_payload(row.limits)
            scope = scope_from_payload(row.scope_name, effective, payload, time.monotonic())
            return Session(
                guard=self,
                name=row.name,
                limits=effective,
                session_id=session_id,
                scope=scope,
                state=row.state,
            )

    # ------------------------------------------------------------------
    # Internal: opening a store and recovering interrupted operations
    # ------------------------------------------------------------------

    def _open_store(self, store: SqliteStore) -> None:
        store.claim(self._agent_name, self._currency, self._instance_id)
        snapshot = store.load()
        now = time.monotonic()
        self._events = list(snapshot.events)
        self._restored_sessions = dict(snapshot.sessions)
        self._restored_payloads = {
            row.scope_name: snapshot.scopes[row.scope_name]
            for row in snapshot.sessions.values()
            if row.scope_name in snapshot.scopes
        }
        agent_payload = snapshot.scopes.get(self._agent_scope.name)
        if agent_payload is not None:
            self._agent_scope = scope_from_payload(
                self._agent_scope.name, self._limits, agent_payload, now
            )
        if snapshot.reservations:
            self._recover_open_reservations(store, snapshot, now)

    def _recover_open_reservations(
        self, store: SqliteStore, snapshot: StoreSnapshot, now: float
    ) -> None:
        """Settle reservations left open by a previous process.

        Each one is treated as a failed operation whose cost is unknown. It is
        charged the larger of its recorded costs and its estimate, so a crash never
        makes spending disappear.
        """
        session_scopes: dict[str, Scope] = {}
        recovered_sessions: set[str] = set()
        for reservation_id, payload in snapshot.reservations:
            session_id = payload["session_id"]
            row = snapshot.sessions.get(session_id)
            if row is None:
                raise PersistenceError(
                    f"reservation {reservation_id} refers to unknown session {session_id}"
                )
            scope = session_scopes.get(session_id)
            if scope is None:
                scope_payload = snapshot.scopes.get(row.scope_name)
                if scope_payload is None:
                    raise PersistenceError(f"session {session_id} has no stored counters")
                scope = scope_from_payload(
                    row.scope_name, limits_from_payload(row.limits), scope_payload, now
                )
                session_scopes[session_id] = scope

            estimated = Decimal(payload["estimated_cost"])
            known = {source: Decimal(amount) for source, amount in payload["known_cost"].items()}
            known_total = sum(known.values(), ZERO)
            charged = max(known_total, estimated)
            breakdown = dict(known)
            if charged > known_total:
                breakdown["estimated_fallback"] = charged - known_total
            apply_settlement(
                [self._agent_scope, scope],
                estimated=estimated,
                charged=charged,
                breakdown=breakdown,
                llm_calls=payload["llm_calls"],
                llm_recorded=payload["llm_recorded"],
                tool_calls=payload["tool_calls"],
                tool_recorded=payload["tool_recorded"],
                success=False,
                elapsed=float(payload["elapsed_seconds"]),
            )
            self._record_event(
                scope=scope.name,
                kind="orphaned_reservation_recovered",
                detail=(
                    f"reservation {reservation_id} was open when the previous process stopped; "
                    f"charged {charged} (estimate {estimated}, recorded {known_total})"
                ),
            )
            recovered_sessions.add(session_id)

        self._restored_payloads.update(
            {scope.name: scope_to_payload(scope, now) for scope in session_scopes.values()}
        )
        scopes = {self._agent_scope.name: scope_to_payload(self._agent_scope, now)}
        scopes.update(
            {scope.name: scope_to_payload(scope, now) for scope in session_scopes.values()}
        )
        try:
            store.save(
                owner=self._instance_id,
                scopes=scopes,
                sessions=[snapshot.sessions[session_id] for session_id in recovered_sessions],
                open_reservations={session_id: [] for session_id in recovered_sessions},
                events=list(self._pending_events),
            )
        except BaseException:
            self._faulted = True
            raise
        self._pending_events.clear()
