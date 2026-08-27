"""Tier 0 -- reconcile by Unique Transaction Reference.

The cheapest and strongest evidence available. When the bank statement carries
a UTR that matches a settlement advice whose payment breakup we hold, there is
nothing to infer: it is a lookup. Roughly a quarter of real credits resolve
here for zero tokens and microseconds of compute, which is precisely why an
LLM has no business being anywhere near this path.

Also the only tier that can catch a duplicate bank posting, because it is the
only one that sees UTRs.
"""

from __future__ import annotations

import re
import time

from ..models import BankTxn
from .base import ESCALATE, EXCEPTION, MATCHED, Decision, MatchContext

# Indian UTRs: bank code, an N/S marker, then a date-and-sequence run.
UTR_RE = re.compile(r"\b([A-Z]{4}[NS][0-9]{10,16})\b")


def extract_utr(txn: BankTxn) -> tuple[str | None, str]:
    if txn.utr:
        return txn.utr, "statement_utr_field"
    m = UTR_RE.search(txn.narration.upper())
    if m:
        return m.group(1), "parsed_from_narration"
    return None, "absent"


class Tier0UTR:
    name = "T0-UTR"

    def attempt(self, txn: BankTxn, ctx: MatchContext) -> Decision:
        t0 = time.perf_counter()

        def done(**kw) -> Decision:
            kw.setdefault("tier", self.name)
            kw["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return Decision(**kw)

        utr, source = extract_utr(txn)
        if not utr:
            return done(outcome=ESCALATE, strategy="no_utr",
                        evidence={"utr_source": source,
                                  "why": "No UTR on the statement line or in the narration."})

        already = utr in ctx.consumed_utrs
        advice = ctx.advice_by_utr.get(utr)

        if already:
            # The bank posted a transfer we have already reconciled. Matching it
            # again would credit the same payments twice and overstate cash.
            return done(
                outcome=EXCEPTION, reason_code="DUPLICATE_UTR", strategy="utr_replay",
                confidence=0.99, settlement_id=advice.settlement_id if advice else None,
                evidence={"utr": utr, "utr_source": source,
                          "why": "This UTR was already reconciled against an earlier "
                                 "credit in this batch. Refusing to post it twice."})

        if advice is None:
            return done(outcome=ESCALATE, strategy="utr_not_in_advices",
                        evidence={"utr": utr, "utr_source": source,
                                  "why": "UTR present but no settlement advice carries it."})

        remaining = ctx.advice_remaining(advice.settlement_id)
        delta = txn.amount_paise - advice.net_paise

        if not advice.breakup_available:
            return done(outcome=ESCALATE, strategy="utr_hit_no_breakup",
                        settlement_id=advice.settlement_id,
                        evidence={"utr": utr, "advice_net_paise": advice.net_paise,
                                  "why": "UTR identifies the settlement but its payment "
                                         "breakup was never downloaded; amount must be "
                                         "decomposed."})

        if abs(delta) > ctx.tolerance_paise:
            return done(outcome=ESCALATE, strategy="utr_hit_amount_mismatch",
                        settlement_id=advice.settlement_id, residual_paise=delta,
                        evidence={"utr": utr, "advice_net_paise": advice.net_paise,
                                  "credit_paise": txn.amount_paise, "delta_paise": delta,
                                  "why": "UTR matches but the amount does not -- likely a "
                                         "split payout or a post-advice adjustment."})

        return done(
            outcome=MATCHED, strategy="utr_direct", confidence=0.99,
            settlement_id=advice.settlement_id, payment_ids=list(advice.payment_ids),
            residual_paise=delta,
            evidence={"utr": utr, "utr_source": source,
                      "advice_net_paise": advice.net_paise,
                      "payment_count": len(advice.payment_ids),
                      "advice_remaining_before_paise": remaining,
                      "why": "UTR on the statement matches a settlement advice whose "
                             "payment breakup is on file and whose net equals the credit."})
