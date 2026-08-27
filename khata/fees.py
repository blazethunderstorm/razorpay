"""Razorpay-shaped MDR + GST model.

The whole reconciliation problem exists because a merchant is credited the
*net* of a payment, never the gross. Getting this arithmetic exactly right --
including the rounding mode -- is what makes subset-sum reconstruction
possible at all. One paise of drift per payment across a 40-payment
settlement is 40 paise of unexplained difference, and 40 paise is enough for
an auditor to reject the whole batch.

Rates below mirror Razorpay's published standard pricing as of 2026. They are
configurable per-merchant because in practice every merchant negotiates.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

GST_RATE = Decimal("0.18")

# method -> MDR as a fraction of gross
DEFAULT_MDR: dict[str, Decimal] = {
    "upi": Decimal("0.0000"),      # UPI is zero-MDR in India
    "card": Decimal("0.0200"),     # 2.00%
    "netbanking": Decimal("0.0190"),  # 1.90%
    "wallet": Decimal("0.0200"),   # 2.00%
}


@dataclass(frozen=True)
class FeeBreakdown:
    gross_paise: int
    mdr_paise: int
    gst_paise: int
    net_paise: int

    @property
    def total_deduction_paise(self) -> int:
        return self.mdr_paise + self.gst_paise


def _round_paise(d: Decimal) -> int:
    return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def compute_fee(gross_paise: int, method: str,
                mdr_table: dict[str, Decimal] | None = None) -> FeeBreakdown:
    """Split a gross payment into MDR, GST-on-MDR, and the net that settles.

    GST is charged on the MDR, not on the transaction value. Both components
    round half-up to the paise independently -- which is what the gateway
    actually does, and therefore what we must replicate bit-for-bit.
    """
    table = mdr_table or DEFAULT_MDR
    rate = table.get(method, Decimal("0.0200"))
    mdr = _round_paise(Decimal(gross_paise) * rate)
    gst = _round_paise(Decimal(mdr) * GST_RATE)
    return FeeBreakdown(
        gross_paise=gross_paise,
        mdr_paise=mdr,
        gst_paise=gst,
        net_paise=gross_paise - mdr - gst,
    )


def net_of(gross_paise: int, method: str,
           mdr_table: dict[str, Decimal] | None = None) -> int:
    return compute_fee(gross_paise, method, mdr_table).net_paise
