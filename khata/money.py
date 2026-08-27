"""Money is paise. Always paise. Never a float.

Every amount in Khata is an ``int`` count of paise (1 rupee = 100 paise).
Floating point is banned from the money path: 0.1 + 0.2 != 0.3 is not a
philosophical curiosity when you are asserting that debits equal credits.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

PAISE = 100


def rupees(amount: float | str | Decimal) -> int:
    """Convert a rupee amount to paise, rounding half-up like a bank does."""
    d = Decimal(str(amount)) * PAISE
    return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fmt(paise: int) -> str:
    """Render paise as Indian-format rupees: 4732811 -> '₹47,328.11'."""
    sign = "-" if paise < 0 else ""
    p = abs(int(paise))
    whole, frac = divmod(p, PAISE)
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        # Indian grouping: last 3 digits, then pairs (12,34,567)
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return f"{sign}₹{s}.{frac:02d}"


def pct(numerator: float, denominator: float) -> float:
    """Percentage that returns 0.0 rather than exploding on an empty batch."""
    return 0.0 if not denominator else round(100.0 * numerator / denominator, 2)
