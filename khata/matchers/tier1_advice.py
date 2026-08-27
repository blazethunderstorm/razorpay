"""Tier 1 -- reconcile by settlement-advice amount.

Plenty of bank statements carry no UTR at all: the narration is a generic
"NEFT CR RAZORPAY SOFTWARE PVT LTD" and nothing more. What we still have is a
settlement advice with an exact net amount and a date. If exactly one open
advice in the window nets to the credit, that is a match on strong evidence.

If *more than one* advice nets to the same amount, this tier refuses and
escalates. Two settlements of identical value in the same week is unusual but
not impossible, and picking one at random would be a coin flip recorded as a
fact.

This tier also handles split payouts, where the bank broke one transfer into
two statement lines and neither line reconciles on its own.
"""

from __future__ import annotations

import time
from datetime import timedelta

from ..models import BankTxn
from .base import ESCALATE, MATCHED, Decision, MatchContext


class Tier1Advice:
    name = "T1-ADVICE"

    def attempt(self, txn: BankTxn, ctx: MatchContext) -> Decision:
        t0 = time.perf_counter()

        def done(**kw) -> Decision:
            kw.setdefault("tier", self.name)
            kw["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return Decision(**kw)

        candidates = ctx.advices_in_window(txn.value_date, with_breakup=True)

        exact = [a for a in candidates
                 if abs(a.net_paise - txn.amount_paise) <= ctx.tolerance_paise
                 and ctx.advice_settled.get(a.settlement_id, 0) == 0]

        if len(exact) == 1:
            a = exact[0]
            return done(
                outcome=MATCHED, strategy="advice_exact_amount", confidence=0.95,
                settlement_id=a.settlement_id, payment_ids=list(a.payment_ids),
                residual_paise=txn.amount_paise - a.net_paise,
                evidence={"advice_id": a.settlement_id, "advice_net_paise": a.net_paise,
                          "advice_date": a.settled_at.date().isoformat(),
                          "payment_count": len(a.payment_ids),
                          "candidates_in_window": len(candidates),
                          "why": "Exactly one unsettled advice in the window nets to this "
                                 "credit to the paise."})

        if len(exact) > 1:
            return done(
                outcome=ESCALATE, strategy="advice_amount_collision",
                evidence={"colliding_advices": [a.settlement_id for a in exact],
                          "why": f"{len(exact)} advices net to this exact amount; "
                                 "amount alone cannot identify the settlement."})

        # --- split payout: this credit plus one sibling line equals an advice ---
        split = self._try_split(txn, ctx, candidates)
        if split is not None:
            return done(**split)

        return done(outcome=ESCALATE, strategy="no_advice_amount_match",
                    evidence={"candidates_in_window": len(candidates),
                              "why": "No open advice nets to this credit, alone or paired."})

    def _try_split(self, txn: BankTxn, ctx: MatchContext, candidates) -> dict | None:
        """Pair this credit with an unmatched sibling landing the same day.

        Both legs are attributed the settlement's full payment set -- the pair
        jointly explains it -- while each leg posts only its own cash to the
        ledger. Attributing half the payments to each leg would be inventing a
        breakup the gateway never issued.
        """
        siblings = [
            t for t in ctx.bank_txns
            if t.bank_txn_id != txn.bank_txn_id
            and t.bank_txn_id not in ctx.matched_txns
            and abs((t.value_date - txn.value_date).days) <= 1
        ]
        for a in candidates:
            need = ctx.advice_remaining(a.settlement_id)
            if need <= 0:
                continue
            # Already-part-paid advice: this leg alone may close it out.
            if (ctx.advice_settled.get(a.settlement_id, 0) > 0
                    and abs(need - txn.amount_paise) <= ctx.tolerance_paise):
                return dict(
                    outcome=MATCHED, strategy="split_leg_closing", confidence=0.93,
                    settlement_id=a.settlement_id, payment_ids=list(a.payment_ids),
                    residual_paise=txn.amount_paise - need,
                    evidence={"advice_id": a.settlement_id,
                              "advice_net_paise": a.net_paise,
                              "already_settled_paise": ctx.advice_settled[a.settlement_id],
                              "this_leg_paise": txn.amount_paise,
                              "why": "Closing leg of a split payout: this credit clears the "
                                     "unsettled remainder of the advice."})
            for s in siblings:
                if abs(txn.amount_paise + s.amount_paise - need) <= ctx.tolerance_paise:
                    return dict(
                        outcome=MATCHED, strategy="split_leg_opening", confidence=0.90,
                        settlement_id=a.settlement_id, payment_ids=list(a.payment_ids),
                        residual_paise=0,
                        evidence={"advice_id": a.settlement_id,
                                  "advice_net_paise": a.net_paise,
                                  "this_leg_paise": txn.amount_paise,
                                  "sibling_txn": s.bank_txn_id,
                                  "sibling_paise": s.amount_paise,
                                  "why": "The bank split one payout across two statement "
                                         "lines; this leg plus its sibling equals the advice "
                                         "net exactly."})
        return None
