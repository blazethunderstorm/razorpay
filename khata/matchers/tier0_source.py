"""Tier 0a -- is this credit a gateway settlement at all?

A merchant's current account receives plenty of money that has nothing to do
with the payment gateway: customer NEFTs, vendor refunds, loan drawdowns,
inter-company sweeps. Asking "which payments make up this credit" before asking
"is this credit a settlement" is how an engine ends up attributing a customer's
direct transfer to twelve unrelated orders. The arithmetic balances, so nothing
looks wrong.

That is not hypothetical -- it is a bug this engine had. A ₹1,62,000 receipt from
a trading company was force-fit into an exact twelve-payment subset at 0.88
confidence, because the non-gateway classification lived in Tier 3 and Tier 2 got
there first. Ordering was the whole defect: the check was already written and
already correct, just asked too late.

The test is deliberately conservative. A genuine settlement either names the
gateway in the narration or carries a UTR we can tie to a settlement advice. Only
a credit with neither is classified out, so an unusual-but-real narration is
escalated rather than discarded.
"""

from __future__ import annotations

import time

from ..models import BankTxn
from .base import ESCALATE, EXCEPTION, Decision, MatchContext
from .tier0_utr import extract_utr

GATEWAY_MARKERS = ("RAZORPAY", "RAZORPAYSO", "RZP")


class Tier0SourceCheck:
    name = "T0-SOURCE"

    def attempt(self, txn: BankTxn, ctx: MatchContext) -> Decision:
        t0 = time.perf_counter()

        def done(**kw) -> Decision:
            kw.setdefault("tier", self.name)
            kw["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return Decision(**kw)

        blob = f"{txn.narration} {txn.counterparty}".upper()
        has_marker = any(m in blob for m in GATEWAY_MARKERS)
        utr, _ = extract_utr(txn)
        utr_known = bool(utr and utr in ctx.advice_by_utr)

        if has_marker or utr_known:
            return done(outcome=ESCALATE, strategy="gateway_credit",
                        evidence={"has_gateway_marker": has_marker,
                                  "utr_matches_an_advice": utr_known,
                                  "why": "Credit references the gateway; proceeding to match."})

        return done(
            outcome=EXCEPTION, reason_code="NOT_A_SETTLEMENT",
            strategy="source_not_gateway", confidence=0.96,
            evidence={"counterparty": txn.counterparty, "narration": txn.narration,
                      "utr": utr,
                      "why": "Neither the narration nor the counterparty references the "
                             "payment gateway, and no UTR ties it to a settlement advice. "
                             "Classified as a direct receipt before any matching is "
                             "attempted -- otherwise an exact subset of unrelated payments "
                             "can be found for it, and would be posted."})
