"""Forward cash forecast -- what is still owed, and when it should land.

Reconciliation looks backwards: this credit, which payments. The same data
answers a question the merchant asks more often, which is *how much is coming
and when*, and it answers it without any inference at all.

Razorpay settles on T+2. So a payment captured on the 20th is paid out on the
22nd. If the last line on the bank statement is dated the 21st, that payment
**cannot** have settled yet -- not "probably hasn't", cannot. The forecast is
therefore arithmetic on the merchant's own capture data, not a prediction, and
it is deliberately restricted to that provable window: anything whose payout
date has already passed is a reconciliation question, not a forecast one, and
belongs in the exception queue where it can be chased.

Refunds and chargebacks are netted off the cycle they are raised in, the same
way the real payout nets them, so the number shown is what the bank will
actually credit rather than gross receivables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .models import Batch

SETTLEMENT_LAG_DAYS = 2


@dataclass
class ForecastDay:
    """One expected payout, built from the payments that will make it up."""

    payout_date: date
    capture_date: date
    payments: int
    gross_paise: int
    mdr_paise: int
    gst_paise: int
    refunds_paise: int
    chargebacks_paise: int

    @property
    def expected_paise(self) -> int:
        """Net of fees, GST, refunds and chargebacks -- what the bank credits."""
        return (self.gross_paise - self.mdr_paise - self.gst_paise
                - self.refunds_paise - self.chargebacks_paise)

    def to_dict(self) -> dict[str, Any]:
        return {
            "payout_date": self.payout_date.isoformat(),
            "capture_date": self.capture_date.isoformat(),
            "payments": self.payments,
            "gross_paise": self.gross_paise,
            "mdr_paise": self.mdr_paise,
            "gst_paise": self.gst_paise,
            "refunds_paise": self.refunds_paise,
            "chargebacks_paise": self.chargebacks_paise,
            "expected_paise": self.expected_paise,
        }


@dataclass
class Forecast:
    as_of: date
    lag_days: int
    days: list[ForecastDay] = field(default_factory=list)

    @property
    def total_paise(self) -> int:
        return sum(d.expected_paise for d in self.days)

    @property
    def payments(self) -> int:
        return sum(d.payments for d in self.days)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "lag_days": self.lag_days,
            "total_paise": self.total_paise,
            "payments": self.payments,
            "days": [d.to_dict() for d in self.days],
        }


def forecast(batch: Batch, *, lag_days: int = SETTLEMENT_LAG_DAYS,
             as_of: date | None = None, horizon: int = 7) -> Forecast:
    """Expected payouts after ``as_of``, which defaults to the merchant's today.

    ``as_of`` defaults to the last day the merchant captured a payment, which is
    the last day they have complete data for -- not the last line on the bank
    statement, which runs two days further ahead precisely because it contains
    the payouts for captures already made.

    Only cycles whose payout date is strictly after ``as_of`` are included. A
    payment whose payout date has already passed has either settled -- the
    reconciler's business -- or is overdue, which is a chase, not a forecast.
    Neither is a claim this function is entitled to make.
    """
    if as_of is None:
        as_of = max((p.captured_at.date() for p in batch.payments), default=date.min)

    buckets: dict[date, ForecastDay] = {}

    def bucket(payout: date, capture: date) -> ForecastDay | None:
        if payout <= as_of or payout > as_of + timedelta(days=horizon):
            return None
        if payout not in buckets:
            buckets[payout] = ForecastDay(payout_date=payout, capture_date=capture,
                                          payments=0, gross_paise=0, mdr_paise=0,
                                          gst_paise=0, refunds_paise=0,
                                          chargebacks_paise=0)
        return buckets[payout]

    for p in batch.payments:
        capture = p.captured_at.date()
        b = bucket(capture + timedelta(days=lag_days), capture)
        if b is None:
            continue
        b.payments += 1
        b.gross_paise += p.gross_paise
        b.mdr_paise += p.mdr_paise
        b.gst_paise += p.gst_paise

    # Refunds and chargebacks net off the cycle they are raised in, not the
    # cycle of the payment they refer to.
    for r in batch.refunds:
        raised = r.created_at.date()
        b = bucket(raised + timedelta(days=lag_days), raised)
        if b is not None:
            b.refunds_paise += r.amount_paise
    for c in batch.chargebacks:
        raised = c.created_at.date()
        b = bucket(raised + timedelta(days=lag_days), raised)
        if b is not None:
            b.chargebacks_paise += c.amount_paise

    return Forecast(as_of=as_of, lag_days=lag_days,
                    days=[buckets[k] for k in sorted(buckets)])
