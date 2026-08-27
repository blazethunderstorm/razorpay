"""Tier 2 -- reconstruct the payment set by exact subset-sum.

This is the tier that exists because settlement breakups go missing. A credit
arrives with no UTR and no advice we can pin it to; all we have is an amount
and a pool of open payments. Recovering which payments it represents is
subset-sum over net-of-fee amounts, solved exactly in ``khata.subsetsum``.

Three passes, each strictly weaker than the last, with the confidence attached
to a match reflecting which pass produced it:

  A. payments only, zero tolerance          -- the clean case
  B. payments plus refunds and chargebacks  -- payouts netted down mid-cycle
  C. payments only, Rs 1 tolerance          -- absorbs rounding drift, and is
                                               deliberately never enough on its
                                               own to clear the posting floor

The guard below matters more than any of them. If a credit sits within a few
hundred rupees of a settlement advice we already hold, the overwhelmingly
likely story is "that settlement, minus an adjustment" -- not "a completely
different set of payments that happens to sum to this". Forcing a subset here
would produce a confident, precise, wrong answer, so the tier steps aside.
"""

from __future__ import annotations

import time

from ..models import BankTxn
from ..money import rupees
from ..subsetsum import Item, solve
from .base import ESCALATE, EXCEPTION, MATCHED, Decision, MatchContext

NEAR_ADVICE_WINDOW = rupees(500)

# Settlement cycles are contractual: a T+2 payout covers payments captured two
# days earlier. Searching that band first is both far faster and far more
# likely to be right than searching everything still open. Widening only
# happens when the tight band genuinely fails.
# A settlement cycle draws from one capture day, not from a rolling window.
# Searching day-by-day keeps every pool small enough for exact enumeration and
# mirrors how the gateway actually forms a payout. T+2 is the contractual lag,
# so it is tried first; the rest fan outward to absorb late or early postings.
DAY_OFFSETS: tuple[int, ...] = (2, 3, 1, 4, 5, 0, 6, 7, 8)

# Fallback bands, used only after every single-day slice has failed. A credit
# that spans capture days is unusual but does happen when a cycle is missed.
FALLBACK_BANDS: tuple[tuple[str, int, int], ...] = (
    ("T+1..T+3 band", 3, 1),
    ("T+0..T+5 band", 5, 0),
)
MAX_EXACT_POOL = 34     # beyond this, exact enumeration stops being affordable


class Tier2Subset:
    name = "T2-SUBSET"

    def attempt(self, txn: BankTxn, ctx: MatchContext) -> Decision:
        t0 = time.perf_counter()

        def done(**kw) -> Decision:
            kw.setdefault("tier", self.name)
            kw["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return Decision(**kw)

        near = self._near_advices(txn, ctx)
        if near:
            return done(
                outcome=ESCALATE, strategy="near_known_advice",
                evidence={"near_advices": near,
                          "why": "Credit sits within Rs 500 of a settlement advice we "
                                 "already hold. Far more likely an adjusted payout than a "
                                 "coincidental subset -- declining to force-fit."})

        credits = ctx.open_credits_in_window(txn.value_date)
        neg = [Item(getattr(c, "refund_id", None) or c.dispute_id, -c.amount_paise)
               for c in credits]
        # A refund or dispute raised on the payout date is netted off that
        # payout -- that is simply how the gateway settles. When one exists we
        # test the netted explanation FIRST. Otherwise a coincidence beats a
        # cause: dropping some unrelated payment that happens to be worth the
        # same as the chargeback also balances, and searching payments-only
        # would find that first and post it with full confidence.
        prefer_netted = any(
            (txn.value_date - c.created_at.date()).days <= 1 for c in credits)

        attempts: list[dict] = []
        oversized = 0

        def search_day(offset: int) -> tuple[object, str, float, int] | None:
            """Run both passes over one capture day. Returns (result, pass, conf, n)."""
            nonlocal oversized
            pool = ctx.open_payments_between(txn.value_date, offset, offset)
            if not pool:
                return None
            if len(pool) > MAX_EXACT_POOL:
                oversized += 1
                attempts.append({"window": f"T+{offset} capture day", "pool": len(pool),
                                 "result": f"skipped, above exact-search cap of {MAX_EXACT_POOL}"})
                return None
            base = [Item(p.payment_id, p.net_paise) for p in pool]
            passes = ([("B_with_credits", base + neg, 0.90), ("A_exact", base, 0.93)]
                      if neg and prefer_netted else
                      [("A_exact", base, 0.93)] + ([("B_with_credits", base + neg, 0.88)]
                                                   if neg else []))
            for pass_name, items, conf in passes:
                if len(items) > MAX_EXACT_POOL:
                    attempts.append({"window": f"T+{offset} capture day", "pool": len(items),
                                     "pass": pass_name,
                                     "result": f"skipped, above exact-search cap of {MAX_EXACT_POOL}"})
                    continue
                res = solve(items, txn.amount_paise, tolerance=0,
                            budget_ms=ctx.subset_budget_ms)
                if res.best is not None and not res.ambiguous:
                    if self._lookahead(res, txn, ctx, items):
                        attempts.append({"window": f"T+{offset} capture day",
                                         "pool": len(items), "pass": pass_name,
                                         "result": "rejected by sibling lookahead"})
                        continue
                if res.ambiguous or res.best is not None:
                    attempts.append({"window": f"T+{offset} capture day", "pool": len(items),
                                     "pass": pass_name,
                                     "result": "ambiguous" if res.ambiguous else "hit",
                                     "strategy": res.strategy})
                    return res, pass_name, conf, offset
                attempts.append({"window": f"T+{offset} capture day", "pool": len(items),
                                 "pass": pass_name, "result": "no subset",
                                 "strategy": res.strategy, "explored": res.explored})
            return None

        # Sweep every candidate capture day rather than stopping at the first
        # hit, including the contractual T+2 one.
        #
        # Stopping early was the source of every false match this engine
        # produced. A pool of ~30 payments has a billion subsets; spread across
        # a few lakh rupees, some subset of a busy day will land on almost any
        # target by coincidence. When a late-posted credit whose real payments
        # sit five days back happens to collide with the T+2 pool, searching
        # T+2 first and trusting it produces a confident, precise, wrong answer
        # -- and the contractual date is exactly the corroboration that makes
        # such an answer look trustworthy.
        #
        # Two days both reconciling means the amount does not identify a cycle.
        # T+2 is then preferred only for *ranking* the survivor, never for
        # overruling a collision.
        hits: list[tuple] = []
        for offset in DAY_OFFSETS:
            if offset > ctx.lookback_days:
                continue
            h = search_day(offset)
            if h is not None:
                hits.append(h)
                if len(hits) >= 2:
                    break

        if len(hits) == 1:
            res, pass_name, conf, off = hits[0]
            tag = "T+2 contractual" if off == 2 else f"T+{off} off-cycle"
            base = conf if off == 2 else min(conf, 0.88)
            if oversized:
                # Some candidate day held more open payments than exact search
                # can enumerate and was skipped. One hit among the days we *did*
                # search is therefore not evidence of uniqueness -- the true
                # capture day may well be one of the days we never opened. This
                # is the day-level form of the same rule the solver applies
                # internally: an incomplete search cannot prove anything, so the
                # match is scored below the posting floor and handed on.
                return done(
                    outcome=ESCALATE, fallback_reason_code="SEARCH_BUDGET_EXCEEDED",
                    strategy=f"unproven[{tag}]", confidence=round(base - 0.20, 2),
                    evidence={"windows_tried": attempts,
                              "days_skipped_as_oversized": oversized,
                              "hit_day": tag,
                              "why": f"A subset was found on {tag}, but {oversized} "
                                     "candidate capture day(s) held too many open payments "
                                     "to enumerate. Uniqueness across days is therefore "
                                     "unproven, so the line detail is not posted."})
            return self._from_result(done, txn, ctx, res, f"{pass_name}[{tag}]",
                                     base, attempts)
        if len(hits) >= 2:
            days = [f"T+{h[3]}" for h in hits]
            return done(
                outcome=ESCALATE, fallback_reason_code="AMBIGUOUS_SUBSET",
                strategy="cross_day_collision", confidence=0.92,
                evidence={"windows_tried": attempts, "colliding_days": days,
                          "why": f"Exact subsets exist on more than one capture day "
                                 f"({', '.join(days)}). The amount does not identify which "
                                 "settlement cycle this credit came from, so the payment "
                                 "breakup is left open and only the cash is attributed."})

        # Multi-day bands, for the rare cycle that genuinely spans capture days.
        for label, lo, hi in FALLBACK_BANDS:
            if lo > ctx.lookback_days:
                continue
            pool = ctx.open_payments_between(txn.value_date, lo, hi)
            if not pool or len(pool) > MAX_EXACT_POOL:
                if pool:
                    oversized += 1
                    attempts.append({"window": label, "pool": len(pool),
                                     "result": f"skipped, above exact-search cap of {MAX_EXACT_POOL}"})
                continue
            base = [Item(p.payment_id, p.net_paise) for p in pool]
            res = solve(base, txn.amount_paise, tolerance=0, budget_ms=ctx.subset_budget_ms)
            if res.ambiguous or res.best is not None:
                attempts.append({"window": label, "pool": len(pool),
                                 "result": "ambiguous" if res.ambiguous else "hit"})
                return self._from_result(done, txn, ctx, res, f"A_exact[{label}]",
                                         0.85, attempts)
            attempts.append({"window": label, "pool": len(pool), "result": "no subset"})

        # Pass C -- Rs 1 tolerance on the two likeliest capture days only.
        # Scored below the posting floor on purpose: an inexact reconciliation
        # is a lead for a human, not a match.
        for d in DAY_OFFSETS[:2]:
            pool = ctx.open_payments_between(txn.value_date, d, d)
            if pool and len(pool) <= MAX_EXACT_POOL:
                res_c = solve([Item(p.payment_id, p.net_paise) for p in pool],
                              txn.amount_paise, tolerance=ctx.loose_tolerance_paise,
                              budget_ms=ctx.subset_budget_ms)
                if res_c.best is not None:
                    attempts.append({"window": f"T+{d} capture day", "pool": len(pool),
                                     "result": "hit within Rs 1 tolerance"})
                    return self._from_result(done, txn, ctx, res_c,
                                             f"C_tolerant[T+{d}]", 0.70, attempts)

        if attempts and oversized == len(attempts):
            return done(outcome=EXCEPTION, reason_code="SEARCH_BUDGET_EXCEEDED",
                        strategy="pool_too_large", confidence=0.60,
                        evidence={"windows": attempts, "cap": MAX_EXACT_POOL,
                                  "why": "Every candidate window held more open payments "
                                         "than exact search can enumerate."})
        if not attempts:
            return done(outcome=EXCEPTION, reason_code="POOL_EXHAUSTED",
                        strategy="empty_pool", confidence=0.95,
                        evidence={"windows": attempts,
                                  "why": "No open payments remain in any candidate settlement window."})

        return done(outcome=EXCEPTION, reason_code="NO_FEASIBLE_SUBSET",
                    strategy="exhausted_all_windows", confidence=0.90,
                    evidence={"windows": attempts,
                              "why": "No subset of the open payments reconciles to this "
                                     "credit in any settlement window, with or without "
                                     "netted refunds."})

    def _lookahead(self, res, txn: BankTxn, ctx: MatchContext,
                   pool_items: list[Item]) -> list[str]:
        """Reject subsets that make a same-day sibling credit unexplainable.

        Several settlement cycles land on the same value date and draw from the
        same capture day. Whichever credit is processed first sees the whole
        day's open payments and can happily consume a subset belonging to its
        neighbour -- the arithmetic balances, so nothing looks wrong, and the
        neighbour is then unmatchable for a reason that has scrolled off the
        screen. That cascade produced every false match this engine made before
        this check existed.

        The test is conditional on purpose: a sibling only counts as stranded
        if it was explainable by this pool *before* we took our subset and is
        provably not explainable after. Orphan credits, which the pool never
        explained, are correctly ignored. And we only act on a *proof* of
        infeasibility -- if the confirming search ran out of budget we say
        nothing rather than reject a good match on a maybe.
        """
        siblings = [
            t for t in ctx.bank_txns
            if t.value_date == txn.value_date
            and t.bank_txn_id != txn.bank_txn_id
            and t.bank_txn_id not in ctx.matched_txns
            and not t.utr                    # UTR-bearing credits never need this pool
        ]
        if not siblings:
            return []

        used = set(res.best.keys)
        remaining = [i for i in pool_items if i.key not in used]
        stranded: list[str] = []
        for sib in siblings[:4]:
            before = solve(pool_items, sib.amount_paise, tolerance=0, budget_ms=300)
            if before.best is None:
                continue                     # never depended on this pool
            after = solve(remaining, sib.amount_paise, tolerance=0, budget_ms=300)
            if after.best is None and after.complete:
                stranded.append(sib.bank_txn_id)
        return stranded

    def _near_advices(self, txn: BankTxn, ctx: MatchContext) -> list[dict]:
        out = []
        for a in ctx.advices_in_window(txn.value_date, with_breakup=True):
            d = txn.amount_paise - a.net_paise
            if 0 < abs(d) <= NEAR_ADVICE_WINDOW:
                out.append({"advice_id": a.settlement_id, "advice_net_paise": a.net_paise,
                            "delta_paise": d})
        return out

    def _from_result(self, done, txn, ctx, res, pass_name, base_conf, attempts):
        best = res.best
        pay_ids = [k for k in best.keys if k.startswith("pay_")]
        ref_ids = [k for k in best.keys if k.startswith("rfnd_")]
        dis_ids = [k for k in best.keys if k.startswith("disp_")]

        shared = {
            "pass": pass_name, "pool_size": res.pool_size, "explored": res.explored,
            "solver_strategy": res.strategy, "solver_ms": res.elapsed_ms,
            "search_complete": res.complete,
            "distinct_decompositions": res.distinct_signatures,
            "payment_count": len(pay_ids),
            "netted_credits": ref_ids + dis_ids,
            "residual_paise": best.residual,
            "windows_tried": attempts,
        }

        if res.ambiguous:
            # Two genuinely different amount decompositions both explain this
            # credit. Posting either one would assert a fact we cannot support.
            alts = [{"payment_count": len(s.keys), "total_paise": s.total}
                    for s in res.solutions[:4]]
            return done(
                outcome=ESCALATE, fallback_reason_code="AMBIGUOUS_SUBSET",
                strategy=res.strategy, confidence=0.95, residual_paise=best.residual,
                payment_ids=[], 
                evidence={**shared, "alternatives": alts,
                          "why": f"{res.distinct_signatures} distinct payment sets reconcile "
                                 "to this credit exactly. The amount cannot tell them apart; "
                                 "posting one would be a guess recorded as a fact. Escalating so the "
                                 "adjudicator can look for a tie-break in the narration."})

        conf = base_conf if res.complete else round(base_conf - 0.18, 2)
        if not res.complete:
            shared["why_downgraded"] = ("Pool too large to enumerate exhaustively, so "
                                        "uniqueness of this decomposition is unproven.")

        return done(
            outcome=MATCHED if conf >= ctx.confidence_floor else ESCALATE,
            strategy=f"subsetsum:{res.strategy}", confidence=conf,
            payment_ids=pay_ids, refund_ids=ref_ids, dispute_ids=dis_ids,
            residual_paise=best.residual,
            evidence={**shared,
                      "why": f"Exact subset of {len(pay_ids)} open payments sums to the "
                             f"credit (pass {pass_name}, {res.strategy}, "
                             f"{res.explored:,} combinations examined)."})
