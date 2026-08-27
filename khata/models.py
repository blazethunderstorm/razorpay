"""Domain objects for settlement reconciliation.

Three source systems feed the engine, and they disagree with each other --
which is the entire point:

  1. Payments export      (the gateway's view of what customers paid)
  2. Settlement advices   (the gateway's view of what it paid out)   -- often incomplete
  3. Bank statement       (the bank's view of what actually landed)

The engine's job is to attribute every rupee of every *bank credit* back to a
set of payments. Ground truth lives on the generator side and is never handed
to a matcher.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class Method(str, Enum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class Scenario(str, Enum):
    """How a given bank credit was constructed. Ground truth only.

    Never exposed to a matcher; used exclusively to score which classes of
    break the engine can and cannot handle, so we can report honestly instead
    of quoting one aggregate number.
    """

    CLEAN_UTR = "CLEAN_UTR"
    MISSING_ADVICE = "MISSING_ADVICE"
    ADVICE_NO_UTR = "ADVICE_NO_UTR"
    BUNDLED_NO_ADVICE = "BUNDLED_NO_ADVICE"
    REFUND_NETTED = "REFUND_NETTED"
    CHARGEBACK_DEBIT = "CHARGEBACK_DEBIT"
    PARTIAL_SPLIT = "PARTIAL_SPLIT"
    TIMING_SKEW = "TIMING_SKEW"
    DUPLICATE_UTR = "DUPLICATE_UTR"
    NARRATION_ONLY = "NARRATION_ONLY"
    ORPHAN_CREDIT = "ORPHAN_CREDIT"
    AMBIGUOUS_SUBSET = "AMBIGUOUS_SUBSET"


@dataclass
class Payment:
    payment_id: str
    order_id: str
    gross_paise: int
    net_paise: int
    mdr_paise: int
    gst_paise: int
    method: str
    captured_at: datetime
    customer_ref: str
    status: str = "captured"  # captured | refunded | disputed

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["captured_at"] = self.captured_at.isoformat()
        return d


@dataclass
class Refund:
    refund_id: str
    payment_id: str
    amount_paise: int
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class Chargeback:
    dispute_id: str
    payment_id: str
    amount_paise: int
    created_at: datetime
    reason: str = "customer_dispute"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class SettlementAdvice:
    """The gateway's payout note.

    ``payment_ids`` is the breakup. In the real world merchants routinely do
    not have it -- they never downloaded the detailed report, the API paginated
    and they stopped, or the settlement predates their integration. When
    ``breakup_available`` is False the engine must reconstruct the set itself.
    """

    settlement_id: str
    utr: str | None
    net_paise: int
    settled_at: datetime
    payment_ids: list[str]
    breakup_available: bool = True
    # Whether the merchant holds this advice at all. Settlement report exports
    # have gaps -- a missed cron, a plan that only retains 90 days, an
    # integration that started mid-quarter. When False the record exists only on
    # the gateway's side and must be discovered before it can be used.
    record_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["settled_at"] = self.settled_at.isoformat()
        if not self.breakup_available:
            d["payment_ids"] = []  # genuinely withheld from the engine
        return d


@dataclass
class BankTxn:
    """One credit line on the merchant's bank statement."""

    bank_txn_id: str
    value_date: date
    amount_paise: int
    narration: str
    utr: str | None
    counterparty: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["value_date"] = self.value_date.isoformat()
        return d


@dataclass
class GroundTruth:
    """What the answer actually is. Held out from every matcher."""

    bank_txn_id: str
    scenario: Scenario
    payment_ids: list[str]
    refund_ids: list[str] = field(default_factory=list)
    dispute_ids: list[str] = field(default_factory=list)
    settlement_id: str | None = None
    resolvable: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scenario"] = self.scenario.value
        return d


@dataclass
class Batch:
    """Everything the engine is allowed to see, plus the sealed answer key.

    ``advices`` is the merchant's own settlement report -- incomplete, and
    mostly without payment-level detail. ``gateway_ledger`` is what Razorpay
    knows: every settlement, with its full breakup. The engine may only reach
    the gateway through the budgeted client in ``khata.gateway``, one call at a
    time, and only for a settlement it has already identified. That asymmetry is
    the real one: the data exists, but pulling all of it is not free.
    """

    batch_id: str
    payments: list[Payment]
    refunds: list[Refund]
    chargebacks: list[Chargeback]
    advices: list[SettlementAdvice]
    bank_txns: list[BankTxn]
    ground_truth: list[GroundTruth]
    gateway_ledger: list[SettlementAdvice] = field(default_factory=list)
    seed: int = 0

    def visible(self) -> dict[str, Any]:
        """The engine's input. Note the absence of ground_truth."""
        return {
            "batch_id": self.batch_id,
            "payments": [p.to_dict() for p in self.payments],
            "refunds": [r.to_dict() for r in self.refunds],
            "chargebacks": [c.to_dict() for c in self.chargebacks],
            "advices": [a.to_dict() for a in self.advices if a.record_available],
            "bank_txns": [b.to_dict() for b in self.bank_txns],
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.visible()
        d["seed"] = self.seed
        d["ground_truth"] = [g.to_dict() for g in self.ground_truth]
        return d
