"""Shared types for the matching tiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

from ..gateway import GatewayClient, NullGateway
from ..models import BankTxn, Chargeback, Payment, Refund, SettlementAdvice

MATCHED = "matched"
ESCALATE = "escalated"
EXCEPTION = "exception"


@dataclass
class Decision:
    tier: str
    outcome: str
    payment_ids: list[str] = field(default_factory=list)
    refund_ids: list[str] = field(default_factory=list)
    dispute_ids: list[str] = field(default_factory=list)
    settlement_id: str | None = None
    # line_level: we know which payments. cash_only: we know which settlement
    # the cash belongs to, but its breakup was never issued and the amount
    # alone does not identify the payments.
    attribution: str = "line_level"
    confidence: float = 0.0
    reason_code: str | None = None
    # What to record if no later tier resolves this. Lets an escalating
    # tier hand forward the diagnosis it already made.
    fallback_reason_code: str | None = None
    residual_paise: int = 0
    strategy: str = ""
    elapsed_ms: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def is_match(self) -> bool:
        return self.outcome == MATCHED


@dataclass
class MatchContext:
    """Mutable engine state shared across tiers for one batch run.

    Bank credits are processed in value-date order, which is what keeps the
    candidate pool small: by the time a late credit is considered, the payments
    belonging to earlier cycles have already been consumed and are no longer
    searchable. Ordering is a performance feature, not an accident.
    """

    payments: dict[str, Payment]
    refunds: dict[str, Refund]
    chargebacks: dict[str, Chargeback]
    advices: list[SettlementAdvice]
    bank_txns: list[BankTxn]

    consumed_payments: set[str] = field(default_factory=set)
    consumed_refunds: set[str] = field(default_factory=set)
    consumed_disputes: set[str] = field(default_factory=set)
    consumed_utrs: set[str] = field(default_factory=set)
    matched_txns: set[str] = field(default_factory=set)
    advice_settled: dict[str, int] = field(default_factory=dict)

    # Budgeted access to the gateway's own settlement records. Defaults to a
    # gateway with no budget, so nothing reaches out unless explicitly enabled.
    gateway: GatewayClient = field(default_factory=NullGateway)

    lookback_days: int = 8
    tolerance_paise: int = 0
    loose_tolerance_paise: int = 100          # Rs 1, only on the fallback pass
    subset_budget_ms: int = 2000
    confidence_floor: float = 0.80            # below this we do not post

    advice_by_utr: dict[str, SettlementAdvice] = field(default_factory=dict)
    advice_by_id: dict[str, SettlementAdvice] = field(default_factory=dict)
    payment_by_order: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a in self.advices:
            if a.utr:
                self.advice_by_utr[a.utr] = a
            self.advice_by_id[a.settlement_id] = a
            self.advice_settled.setdefault(a.settlement_id, 0)
        for p in self.payments.values():
            self.payment_by_order[p.order_id] = p.payment_id

    # ---- availability ----

    def advice_remaining(self, sid: str) -> int:
        a = self.advice_by_id[sid]
        return a.net_paise - self.advice_settled.get(sid, 0)

    def open_payments_in_window(self, value_date: date) -> list[Payment]:
        lo = value_date - timedelta(days=self.lookback_days)
        return [
            p for p in self.payments.values()
            if p.payment_id not in self.consumed_payments
            and lo <= p.captured_at.date() <= value_date
        ]

    def open_payments_between(self, value_date: date, lo_days: int,
                              hi_days: int) -> list[Payment]:
        """Open payments captured between value_date-lo_days and value_date-hi_days."""
        lo = value_date - timedelta(days=lo_days)
        hi = value_date - timedelta(days=hi_days)
        return [
            p for p in self.payments.values()
            if p.payment_id not in self.consumed_payments
            and lo <= p.captured_at.date() <= hi
        ]

    def open_credits_in_window(self, value_date: date) -> list[Refund | Chargeback]:
        """Refunds and chargebacks that could have been netted off a payout."""
        lo = value_date - timedelta(days=1)
        out: list[Refund | Chargeback] = []
        for r in self.refunds.values():
            if r.refund_id not in self.consumed_refunds and lo <= r.created_at.date() <= value_date:
                out.append(r)
        for c in self.chargebacks.values():
            if c.dispute_id not in self.consumed_disputes and lo <= c.created_at.date() <= value_date:
                out.append(c)
        return out

    def advices_in_window(self, value_date: date, *, with_breakup: bool | None = None,
                          unconsumed_only: bool = True) -> list[SettlementAdvice]:
        lo = value_date - timedelta(days=self.lookback_days)
        out = []
        for a in self.advices:
            d = a.settled_at.date()
            if not (lo <= d <= value_date + timedelta(days=1)):
                continue
            if with_breakup is not None and a.breakup_available != with_breakup:
                continue
            if unconsumed_only and self.advice_remaining(a.settlement_id) <= 0:
                continue
            out.append(a)
        return out

    def register_advice(self, advice: SettlementAdvice) -> None:
        """Adopt a settlement discovered from the gateway into local state."""
        if advice.settlement_id in self.advice_by_id:
            return
        self.advices.append(advice)
        self.advice_by_id[advice.settlement_id] = advice
        if advice.utr:
            self.advice_by_utr.setdefault(advice.utr, advice)
        self.advice_settled.setdefault(advice.settlement_id, 0)

    def mark_breakup_recovered(self, settlement_id: str, payment_ids: list[str]) -> None:
        """Record a breakup bought from the gateway so a later leg reuses it
        instead of spending a second call on the same settlement."""
        a = self.advice_by_id.get(settlement_id)
        if a is not None:
            a.payment_ids = list(payment_ids)
            a.breakup_available = True

    # ---- consumption ----

    def consume(self, decision: Decision, txn: BankTxn) -> None:
        for pid in decision.payment_ids:
            self.consumed_payments.add(pid)
        for rid in decision.refund_ids:
            self.consumed_refunds.add(rid)
        for did in decision.dispute_ids:
            self.consumed_disputes.add(did)
        if txn.utr:
            self.consumed_utrs.add(txn.utr)
        if decision.settlement_id:
            self.advice_settled[decision.settlement_id] = (
                self.advice_settled.get(decision.settlement_id, 0) + txn.amount_paise)
        self.matched_txns.add(txn.bank_txn_id)


class Tier(Protocol):
    name: str

    def attempt(self, txn: BankTxn, ctx: MatchContext) -> Decision: ...
