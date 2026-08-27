"""Tier 2b -- attribute the cash, then buy back the line detail if it is worth it.

Tier 2 proves, by exhaustive enumeration, when a credit's payment set cannot be
recovered from the amount: either more than one decomposition exists, or none
does. At that point the merchant's own data is out of answers. But the merchant
still needs their bank balance explained, and two further moves are available.

**Attribute the cash.** If exactly one settlement advice nets to this credit, the
cash belongs to that settlement whatever we know about its lines. That is posted
as ``cash_only``.

**Then buy the lines.** The gateway *does* hold the payment breakup, behind a
rate-limited endpoint that returns one settlement at a time. So once a
settlement is identified we spend one call to recover its lines and upgrade the
attribution from ``cash_only`` to ``line_level``.

Two constraints make that defensible rather than a shortcut:

The call is only made **after** the settlement has been identified by our own
arithmetic. The endpoint answers "which payments", never "which settlement" --
point it at the wrong settlement and it returns the wrong payments with total
confidence. Attribution precision still carries the whole load.

The call is **budgeted**, and the budget is small on purpose. A quarter of
history is thousands of settlements; pulling all of them is exactly the thing
merchants cannot do, which is why this problem is solved by arithmetic first and
by API calls only where arithmetic provably cannot reach. When the budget runs
out the tier degrades to plain cash attribution rather than failing.

The same client also **discovers** settlements the merchant's own report never
captured -- the one case where no amount lookup can help, because the settlement
has no local record to look up.
"""

from __future__ import annotations

import time
from datetime import datetime

from ..models import BankTxn, SettlementAdvice
from .base import ESCALATE, MATCHED, Decision, MatchContext


class Tier2bCashOnly:
    name = "T2B-CASH"

    def attempt(self, txn: BankTxn, ctx: MatchContext) -> Decision:
        t0 = time.perf_counter()

        def done(**kw) -> Decision:
            kw.setdefault("tier", self.name)
            kw["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return Decision(**kw)

        source = "merchant_settlement_report"
        candidates = [
            a for a in ctx.advices_in_window(txn.value_date)
            if abs(a.net_paise - txn.amount_paise) <= ctx.tolerance_paise
            and ctx.advice_settled.get(a.settlement_id, 0) == 0
        ]
        discovered: list[dict] = []

        # No local record nets to this credit. Before giving up, ask the gateway
        # what it settled that day -- the merchant's export may simply be missing
        # the row. This costs one call and returns no payment detail.
        if not candidates and ctx.gateway.remaining > 0:
            listed = ctx.gateway.list_settlements(txn.value_date, txn.bank_txn_id)
            known = {a.settlement_id for a in ctx.advices}
            discovered = [
                s for s in listed
                if s["settlement_id"] not in known
                and abs(s["net_paise"] - txn.amount_paise) <= ctx.tolerance_paise
                and ctx.advice_settled.get(s["settlement_id"], 0) == 0
            ]
            if len(discovered) == 1:
                s = discovered[0]
                candidates = [SettlementAdvice(
                    settlement_id=s["settlement_id"], utr=s["utr"],
                    net_paise=s["net_paise"], settled_at=s["settled_at"],
                    payment_ids=[], breakup_available=False, record_available=False)]
                source = "gateway_settlement_list"
                ctx.register_advice(candidates[0])

        if len(candidates) != 1:
            return done(
                outcome=ESCALATE, strategy="no_unique_settlement_by_amount",
                evidence={"local_candidates": [a.settlement_id for a in candidates],
                          "discovered_candidates": len(discovered),
                          "gateway_calls_remaining": ctx.gateway.remaining,
                          "why": ("No settlement nets to this credit, in the merchant's "
                                  "report or on the gateway."
                                  if not candidates else
                                  f"{len(candidates)} settlements net to this exact amount; "
                                  "cannot attribute the cash to one of them.")})

        a = candidates[0]
        ev = {
            "advice_id": a.settlement_id, "advice_net_paise": a.net_paise,
            "advice_date": a.settled_at.date().isoformat(),
            "identified_from": source,
            "breakup_in_merchant_report": a.breakup_available,
        }

        # The settlement is identified. Spend one call to recover its lines.
        # Under the "value" policy the gateway is holding: the request is parked
        # here and replayed after the batch, largest credit first.
        if not a.breakup_available and ctx.gateway.remaining > 0:
            pids = ctx.gateway.fetch_recon(a.settlement_id, txn.bank_txn_id)
            if pids:
                fresh = [p for p in pids if p not in ctx.consumed_payments]
                total = sum(ctx.payments[p].net_paise for p in fresh
                            if p in ctx.payments)
                ev.update({
                    "gateway_recon": "hit", "recon_payments": len(pids),
                    "recon_unconsumed": len(fresh),
                    "recon_net_paise": total,
                    "recon_delta_paise": total - txn.amount_paise,
                })
                # Never trust a fetch blind either: the breakup's own nets must
                # reconcile to the credit, allowing for refunds and chargebacks
                # netted off the payout in the same cycle.
                netting = sum(c.amount_paise for c in
                              ctx.open_credits_in_window(txn.value_date))
                reconciles = (abs(total - txn.amount_paise) <= ctx.tolerance_paise
                              or 0 <= total - txn.amount_paise <= netting)
                if fresh and reconciles:
                    ctx.mark_breakup_recovered(a.settlement_id, pids)
                    return done(
                        outcome=MATCHED, strategy=f"gateway_recon[{source}]",
                        confidence=0.96, attribution="line_level",
                        settlement_id=a.settlement_id, payment_ids=fresh,
                        residual_paise=total - txn.amount_paise,
                        evidence={**ev, "why":
                            "Settlement identified from the credit amount, then its payment "
                            "breakup recovered with one gateway call. The recovered nets "
                            "reconcile to the credit, so the line detail is posted rather "
                            "than left open."})
                ev["gateway_recon"] = "fetched but did not reconcile"
            elif ctx.gateway.hold:
                ev["gateway_recon"] = "held for the value pass"
            else:
                ev["gateway_recon"] = ("budget exhausted" if ctx.gateway.remaining <= 0
                                       else "miss")

        return done(
            outcome=MATCHED, strategy=f"advice_amount_cash_only[{source}]",
            confidence=0.91, attribution="cash_only",
            settlement_id=a.settlement_id, payment_ids=[],
            residual_paise=txn.amount_paise - a.net_paise,
            evidence={**ev, "payment_breakup": "not recovered",
                      "why": "Exactly one settlement nets to this credit, so the cash is "
                             "attributed with confidence. Its payment breakup is neither in "
                             "the merchant's report nor recoverable within the gateway call "
                             "budget, so the line detail is left open rather than guessed."})
