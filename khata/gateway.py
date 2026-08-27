"""Budgeted client for the gateway's own settlement records.

Razorpay does expose the payment-level breakup of a settlement -- but as a
separate, paginated, rate-limited endpoint, one settlement at a time. A merchant
reconciling a quarter of history cannot simply pull everything, which is exactly
why reconciliation is done by arithmetic in the first place.

So this client is deliberately expensive and deliberately gated:

* Every call is counted against a budget and refused past it.
* Every call is recorded with the credit that justified it, so a reviewer can
  ask "why did we spend that call" and get an answer.
* ``fetch_recon`` only works for a settlement id the engine has **already
  identified by its own means**. It resolves *which payments*, never *which
  settlement*. Point it at the wrong settlement and it returns the wrong
  payments with total confidence -- so attribution precision still carries the
  whole load.

That last property is what keeps the metrics meaningful. This is a data source,
not an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .models import SettlementAdvice


@dataclass
class GatewayCall:
    op: str
    argument: str
    on_behalf_of: str
    hit: bool
    note: str = ""


@dataclass
class HeldRequest:
    """A recon fetch Tier 2b wanted, parked until the budget can be aimed."""

    settlement_id: str
    on_behalf_of: str


@dataclass
class GatewayClient:
    """Simulates the gateway's settlement endpoints under a hard call budget."""

    ledger: list[SettlementAdvice]
    budget: int = 40

    # Deferred allocation. The budget is the scarce resource in this system --
    # the ablation puts these calls at +38 points of line recall, more than any
    # other capability -- and spending it first-come-first-served means a small
    # credit early in the month can burn the call a large one later needs.
    #
    # While ``hold`` is set, ``fetch_recon`` records the request and returns
    # None without charging. The engine then replays the held requests largest
    # credit first, until the budget is gone. A held request is not a refusal:
    # every call is still spent, just aimed better.
    hold: bool = False
    held: list[HeldRequest] = field(default_factory=list)

    calls: list[GatewayCall] = field(default_factory=list)
    _by_id: dict[str, SettlementAdvice] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {a.settlement_id: a for a in self.ledger}

    # ---- budget ----

    @property
    def spent(self) -> int:
        return len(self.calls)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    def _charge(self, op: str, arg: str, who: str, hit: bool, note: str = "") -> bool:
        if self.remaining <= 0:
            self.calls.append(GatewayCall(op, arg, who, False, "budget exhausted"))
            return False
        self.calls.append(GatewayCall(op, arg, who, hit, note))
        return True

    # ---- endpoints ----

    def fetch_recon(self, settlement_id: str, on_behalf_of: str) -> list[str] | None:
        """GET /settlements/{id}/recon -- the payment-level breakup.

        Returns None when the budget is spent or the id is unknown to the
        gateway. A None is a real outcome, not an error: the caller must be able
        to carry on with whatever it already had.
        """
        if self.hold:
            self.held.append(HeldRequest(settlement_id, on_behalf_of))
            return None
        a = self._by_id.get(settlement_id)
        if a is None:
            self._charge("fetch_recon", settlement_id, on_behalf_of, False, "unknown id")
            return None
        if self.remaining <= 0:
            self._charge("fetch_recon", settlement_id, on_behalf_of, False)
            return None
        self._charge("fetch_recon", settlement_id, on_behalf_of, True,
                     f"{len(a.payment_ids)} payments")
        return list(a.payment_ids)

    def list_settlements(self, on_date: date, on_behalf_of: str) -> list[dict[str, Any]]:
        """GET /settlements?from=&to= -- ids, nets and dates, no breakups.

        This is how a settlement the merchant never exported gets discovered at
        all. It returns no payment detail, so discovering a settlement here still
        leaves the line-level question open.
        """
        if self.remaining <= 0:
            self._charge("list_settlements", on_date.isoformat(), on_behalf_of, False)
            return []
        found = [a for a in self.ledger if a.settled_at.date() == on_date]
        self._charge("list_settlements", on_date.isoformat(), on_behalf_of, True,
                     f"{len(found)} settlements")
        return [{"settlement_id": a.settlement_id, "net_paise": a.net_paise,
                 "utr": a.utr, "settled_at": a.settled_at} for a in found]

    # ---- reporting ----

    def summary(self) -> dict[str, Any]:
        by_op: dict[str, int] = {}
        for c in self.calls:
            by_op[c.op] = by_op.get(c.op, 0) + 1
        return {
            "budget": self.budget, "spent": self.spent, "remaining": self.remaining,
            "by_op": by_op,
            "hits": sum(c.hit for c in self.calls),
            "refused": sum(1 for c in self.calls if c.note == "budget exhausted"),
            "held_requests": len(self.held),
            "calls": [c.__dict__ for c in self.calls],
        }


class NullGateway(GatewayClient):
    """A gateway with no budget at all -- used for the offline ablation."""

    def __init__(self) -> None:
        super().__init__(ledger=[], budget=0)
