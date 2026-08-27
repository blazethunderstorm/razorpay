"""Synthetic settlement data with a sealed answer key.

Design rule: the generator and the matchers share *no* code beyond the fee
model. If the generator's idea of "how a settlement is built" were reused by
the matcher, the reported match rate would be measuring nothing but our own
consistency. The fee model is the one deliberate exception -- both the gateway
and the merchant genuinely compute MDR the same way, so sharing it is
faithful, not circular.

Every bank credit is stamped with the ``Scenario`` that produced it so results
can be broken down by failure class instead of collapsing into one number that
hides where the engine is weak.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta

from .fees import compute_fee
from .models import (
    BankTxn, Batch, Chargeback, GroundTruth, Payment, Refund, Scenario,
    SettlementAdvice,
)
from .money import rupees

BANKS = ["HDFC", "ICIC", "UTIB", "SBIN", "KKBK"]
ORPHAN_SENDERS = [
    "ACME RETAIL PVT LTD", "SUNRISE TRADERS", "VENKATESH ENTERPRISES",
    "BLUEDART LOGISTICS", "MEHTA & SONS HUF", "KAVERI FOODS LLP",
]
METHOD_WEIGHTS = [("upi", 0.45), ("card", 0.30), ("netbanking", 0.15), ("wallet", 0.10)]

# Scenario mix for settlement-backed credits. Tuned so every break class gets
# enough instances to produce a meaningful per-scenario recall number rather
# than a sample of one.
SCENARIO_WEIGHTS = [
    (Scenario.CLEAN_UTR, 0.22),
    (Scenario.MISSING_ADVICE, 0.10),
    (Scenario.ADVICE_NO_UTR, 0.12),
    (Scenario.BUNDLED_NO_ADVICE, 0.17),
    (Scenario.REFUND_NETTED, 0.11),
    (Scenario.CHARGEBACK_DEBIT, 0.06),
    (Scenario.PARTIAL_SPLIT, 0.07),
    (Scenario.TIMING_SKEW, 0.06),
    (Scenario.NARRATION_ONLY, 0.09),
    (Scenario.DUPLICATE_UTR, 0.03),
]


class Generator:
    def __init__(self, seed: int = 42, n_payments: int = 600, days: int = 21,
                 orphan_rate: float = 0.06, ambiguous_credits: int = 2):
        self.rng = random.Random(seed)
        self.seed = seed
        self.n_payments = n_payments
        self.days = days
        self.orphan_rate = orphan_rate
        self.ambiguous_credits = ambiguous_credits
        self._seq = 0
        self.start = date(2026, 7, 1)

    # ---------- id helpers ----------

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def _pid(self) -> str:
        return f"pay_{self._next():012d}"

    def _utr(self, d: date) -> str:
        bank = self.rng.choice(BANKS)
        return f"{bank}N{d.strftime('%y%m%d')}{self.rng.randint(100000, 999999)}"

    # ---------- primitives ----------

    def _amount(self) -> int:
        """E-commerce-shaped ticket sizes with realistic price points."""
        r = self.rng.random()
        if r < 0.60:
            base = self.rng.choice([199, 249, 299, 349, 399, 499, 599, 699, 799, 899,
                                    999, 1099, 1249, 1399, 1499])
        elif r < 0.90:
            base = self.rng.randrange(1500, 8000, 50) + self.rng.choice([0, 9, 49, 99])
        else:
            base = self.rng.randrange(8000, 60000, 100) + self.rng.choice([0, 99])
        return rupees(base)

    def _method(self) -> str:
        r, acc = self.rng.random(), 0.0
        for m, w in METHOD_WEIGHTS:
            acc += w
            if r <= acc:
                return m
        return "card"

    def _scenario(self) -> Scenario:
        r, acc = self.rng.random(), 0.0
        total = sum(w for _, w in SCENARIO_WEIGHTS)
        for s, w in SCENARIO_WEIGHTS:
            acc += w / total
            if r <= acc:
                return s
        return Scenario.CLEAN_UTR

    def _make_payment(self, when: datetime) -> Payment:
        gross = self._amount()
        method = self._method()
        fb = compute_fee(gross, method)
        pid = self._pid()
        return Payment(
            payment_id=pid,
            order_id=f"order_{pid[4:]}",
            gross_paise=fb.gross_paise,
            net_paise=fb.net_paise,
            mdr_paise=fb.mdr_paise,
            gst_paise=fb.gst_paise,
            method=method,
            captured_at=when,
            customer_ref=f"cust_{self.rng.randint(10000, 99999)}",
        )

    # ---------- narration templates ----------

    def _narration_with_utr(self, utr: str) -> str:
        return self.rng.choice([
            f"NEFT-RAZORPAY SOFTWARE PVT LTD-{utr}-MERCHANT SETTLEMENT",
            f"IMPS/{utr}/RAZORPAY/SETTLEMENT CR",
            f"RTGS CR {utr} RAZORPAY SOFTWARE PRIVATE LIMITED",
            f"UPI/CR/{utr}/RAZORPAYSO/PAYOUT",
        ])

    def _narration_no_utr(self) -> str:
        return self.rng.choice([
            "NEFT CR-HDFC0000060-RAZORPAY SOFTWARE PRIVATE LIMITED-MERCHANT PAYOUT",
            "ACH C- RAZORPAY SOFTWARE PVT LTD -SETTLEMENT",
            "BY TRANSFER-NEFT*RAZORPAY SOFTWARE PVT*PAYOUT",
        ])

    def _narration_reference_only(self, advice: SettlementAdvice,
                                  payments: list[Payment]) -> tuple[str, str]:
        """Free-text credit whose only usable clue is buried in the narration.

        Two flavours, deliberately different in difficulty:

        ``settlement_ref`` embeds the settlement id -- a determined regex can
        recover it, which is exactly what the offline fallback does.

        ``order_ref`` names *order* ids, not payment ids. Recovering the match
        requires joining orders to payments, which no regex over the narration
        can do on its own. This is the flavour that genuinely needs Tier 3.
        """
        if self.rng.random() < 0.45:
            return (f"RTGS CR RAZORPAY SETTLEMENT {advice.settlement_id.upper()} "
                    f"LESS GATEWAY ADJ", "settlement_ref")
        sample = payments[:3]
        refs = ",".join(p.order_id.replace("order_", "").lstrip("0") for p in sample)
        return (f"NEFT RAZORPAY MERCHANT PAYOUT ORD {refs} AND OTHERS LESS ADJ",
                "order_ref")

    # ---------- main build ----------

    def _scenario_schedule(self, n: int) -> list[Scenario]:
        """Stratified scenario plan rather than an independent draw per credit.

        A pure random draw at this sample size routinely produces zero
        instances of the rarer break classes, which would let the engine post a
        flattering score for scenarios it was never actually tested on.
        Stratifying guarantees every class is exercised.
        """
        total_w = sum(w for _, w in SCENARIO_WEIGHTS)
        sched: list[Scenario] = []
        for s, w in SCENARIO_WEIGHTS:
            sched.extend([s] * max(1, round(n * w / total_w)))
        while len(sched) < n:
            sched.append(Scenario.CLEAN_UTR)
        self.rng.shuffle(sched)
        return sched[:n]

    def build(self) -> Batch:
        payments: list[Payment] = []
        refunds: list[Refund] = []
        chargebacks: list[Chargeback] = []
        advices: list[SettlementAdvice] = []
        bank_txns: list[BankTxn] = []
        truth: list[GroundTruth] = []

        by_day: dict[date, list[Payment]] = {}
        per_day = max(6, self.n_payments // self.days)

        for day_idx in range(self.days):
            d = self.start + timedelta(days=day_idx)
            todays: list[Payment] = []
            # Weekend dip -- uneven daily volume stops the subset-sum search
            # from having one uniform pool shape to exploit.
            count = per_day if d.weekday() < 5 else max(4, int(per_day * 0.55))
            for _ in range(count):
                when = datetime.combine(d, time(self.rng.randint(0, 23),
                                                self.rng.randint(0, 59),
                                                self.rng.randint(0, 59)))
                p = self._make_payment(when)
                todays.append(p)
                payments.append(p)
            by_day[d] = todays

        # Refunds (~5%) and chargebacks (~1.2%), raised a day or two after capture.
        for p in payments:
            r = self.rng.random()
            if r < 0.05:
                refunds.append(Refund(
                    refund_id=f"rfnd_{self._next():012d}",
                    payment_id=p.payment_id,
                    amount_paise=p.gross_paise,
                    created_at=p.captured_at + timedelta(days=self.rng.randint(1, 3)),
                ))
                p.status = "refunded"
            elif r < 0.062:
                chargebacks.append(Chargeback(
                    dispute_id=f"disp_{self._next():012d}",
                    payment_id=p.payment_id,
                    amount_paise=p.gross_paise,
                    created_at=p.captured_at + timedelta(days=self.rng.randint(2, 5)),
                ))
                p.status = "disputed"

        refunds_by_day: dict[date, list[Refund]] = {}
        for r in refunds:
            refunds_by_day.setdefault(r.created_at.date(), []).append(r)
        cb_by_day: dict[date, list[Chargeback]] = {}
        for c in chargebacks:
            cb_by_day.setdefault(c.created_at.date(), []).append(c)

        pay_by_id = {p.payment_id: p for p in payments}

        # Settlement plan: T+2, with 1-3 cycles a day. Multiple daily cycles are
        # standard for higher-volume merchants and they are what makes the
        # per-credit candidate pool small enough to search exactly.
        plan: list[tuple[date, int, int]] = []
        for day_idx in range(self.days):
            capture_day = self.start + timedelta(days=day_idx)
            n_cycles = self.rng.choice([1, 2, 2, 3])
            for c in range(n_cycles):
                plan.append((capture_day, c, n_cycles))
        schedule = self._scenario_schedule(len(plan))

        used_utr_dates: dict[str, date] = {}

        for (capture_day, cycle, n_cycles), scenario in zip(plan, schedule):
            settle_day = capture_day + timedelta(days=2)
            day_pool = by_day[capture_day]
            # Slice this cycle's share of the day's captures.
            size = len(day_pool) // n_cycles
            lo = cycle * size
            hi = len(day_pool) if cycle == n_cycles - 1 else lo + size
            pool = day_pool[lo:hi]
            if len(pool) < 3:
                continue

            # A cycle usually carries its whole slice minus a couple of
            # stragglers that slip to the next window. Sometimes it carries far
            # less, which is what pushes the solver off its fast paths.
            if self.rng.random() < 0.70:
                drop = self.rng.randint(0, min(3, len(pool) - 2))
            else:
                drop = self.rng.randint(3, max(3, int(len(pool) * 0.45)))
            shuffled = list(pool)
            self.rng.shuffle(shuffled)
            chosen = shuffled[drop:]
            if len(chosen) < 2:
                continue

            gross_net = sum(p.net_paise for p in chosen)
            netted_refunds: list[Refund] = []
            netted_cbs: list[Chargeback] = []

            if scenario is Scenario.REFUND_NETTED:
                cands = [r for r in refunds_by_day.get(settle_day, [])
                         if r.payment_id in pay_by_id][:2]
                netted_refunds = cands
                gross_net -= sum(r.amount_paise for r in cands)
            elif scenario is Scenario.CHARGEBACK_DEBIT:
                cands = cb_by_day.get(settle_day, [])[:1]
                netted_cbs = cands
                gross_net -= sum(c.amount_paise for c in cands)

            if gross_net <= 0:
                continue

            sid = f"setl_{self._next():010d}"
            utr = self._utr(settle_day)
            breakup = scenario in (Scenario.CLEAN_UTR, Scenario.ADVICE_NO_UTR,
                                   Scenario.NARRATION_ONLY, Scenario.DUPLICATE_UTR,
                                   Scenario.PARTIAL_SPLIT)
            # MISSING_ADVICE: the settlement happened and the gateway knows all
            # about it, but the merchant's own settlement report never captured
            # it. Nothing in the merchant's data names this settlement, so no
            # amount lookup and no API fetch can reach it -- the payment set has
            # to be reconstructed from the credit amount alone.
            record = scenario is not Scenario.MISSING_ADVICE
            advice = SettlementAdvice(
                settlement_id=sid, utr=utr, net_paise=gross_net,
                settled_at=datetime.combine(settle_day, time(18, 30)),
                payment_ids=[p.payment_id for p in chosen],
                breakup_available=breakup and record,
                record_available=record,
            )
            advices.append(advice)

            value_date = settle_day
            if scenario is Scenario.TIMING_SKEW:
                value_date = settle_day + timedelta(days=self.rng.choice([3, 4, 5]))

            def _emit(txn_amount: int, narration: str, txn_utr: str | None,
                      sc: Scenario, pids: list[str], vdate: date,
                      resolvable: bool = True, note: str = "",
                      _rf=netted_refunds, _cb=netted_cbs, _sid=sid) -> None:
                bt = BankTxn(
                    bank_txn_id=f"bnk_{self._next():010d}",
                    value_date=vdate, amount_paise=txn_amount, narration=narration,
                    utr=txn_utr, counterparty="RAZORPAY SOFTWARE PVT LTD",
                )
                bank_txns.append(bt)
                truth.append(GroundTruth(
                    bank_txn_id=bt.bank_txn_id, scenario=sc, payment_ids=pids,
                    refund_ids=[r.refund_id for r in _rf],
                    dispute_ids=[c.dispute_id for c in _cb],
                    settlement_id=_sid, resolvable=resolvable, note=note,
                ))

            pids = [p.payment_id for p in chosen]

            if scenario is Scenario.PARTIAL_SPLIT:
                # One payout, two statement lines -- the bank split the transfer.
                # Neither leg reconciles alone; only the pair does.
                half = gross_net // 2
                u2 = self._utr(value_date)
                _emit(half, self._narration_with_utr(utr) + " 1/2", utr,
                      scenario, pids, value_date,
                      note="Leg 1 of 2 -- settlement split across two bank credits")
                _emit(gross_net - half, self._narration_with_utr(u2) + " 2/2", u2,
                      scenario, pids, value_date,
                      note="Leg 2 of 2 -- settlement split across two bank credits")
                continue

            if scenario is Scenario.NARRATION_ONLY:
                # A small unexplained gateway adjustment makes the arithmetic
                # tiers fail by construction; the only remaining signal is prose.
                adj = rupees(self.rng.choice([17, 23, 47, 61, 89]))
                narration, flavour = self._narration_reference_only(advice, chosen)
                _emit(gross_net - adj, narration, None, scenario, pids, value_date,
                      note=f"Unexplained gateway adjustment; narration clue flavour={flavour}")
                continue

            if scenario is Scenario.DUPLICATE_UTR:
                _emit(gross_net, self._narration_with_utr(utr), utr,
                      scenario, pids, value_date)
                # The bank posted the same transfer twice. The repost must NOT
                # be matched to the same payments -- double-counting cash is a
                # worse error than leaving a line open.
                _emit(gross_net, self._narration_with_utr(utr) + " (REPOST)", utr,
                      scenario, [], value_date, resolvable=False,
                      note="Duplicate bank posting of an already-reconciled UTR")
                continue

            if scenario is Scenario.MISSING_ADVICE:
                _emit(gross_net, self._narration_no_utr(), None, scenario,
                      pids, value_date,
                      note="Settlement exists on the gateway but not in the "
                           "merchant's settlement report; reconstruct from amount only")
                continue

            if scenario is Scenario.CLEAN_UTR:
                _emit(gross_net, self._narration_with_utr(utr), utr, scenario,
                      pids, value_date)
            elif scenario is Scenario.ADVICE_NO_UTR:
                _emit(gross_net, self._narration_no_utr(), None, scenario,
                      pids, value_date)
            else:
                _emit(gross_net, self._narration_no_utr(), None, scenario,
                      pids, value_date)
            used_utr_dates[utr] = value_date

        self._inject_ambiguous(payments, by_day, bank_txns, truth, set())
        self._inject_orphans(bank_txns, truth)

        bank_txns.sort(key=lambda b: (b.value_date, b.bank_txn_id))
        return Batch(
            batch_id=f"batch_{self.seed}", payments=payments, refunds=refunds,
            chargebacks=chargebacks,
            # What the merchant actually holds vs what the gateway knows.
            advices=[a for a in advices if a.record_available],
            gateway_ledger=advices,
            bank_txns=bank_txns, ground_truth=truth, seed=self.seed,
        )

    def _inject_ambiguous(self, payments: list[Payment],
                          by_day: dict[date, list[Payment]],
                          bank_txns: list[BankTxn], truth: list[GroundTruth],
                          settled: set[str]) -> None:
        """Plant credits that two different payment sets both explain exactly.

        UPI carries zero MDR, so net equals gross and equal-sum subsets stay
        equal after fees. A solver that returns the first subset it finds will
        be wrong roughly half the time here and will never know it. The correct
        behaviour is to detect the collision and refuse to post.
        """
        for i in range(self.ambiguous_credits):
            d = self.start + timedelta(days=self.days - 3 - i)
            when = datetime.combine(d, time(11, 0))
            a, b, c = rupees(1200), rupees(1800), rupees(3000)
            x, y = rupees(2500), rupees(3500)   # a+b+c == x+y == 6000
            group: list[Payment] = []
            for amt in (a, b, c, x, y):
                fb = compute_fee(amt, "upi")
                pid = self._pid()
                p = Payment(
                    payment_id=pid, order_id=f"order_{pid[4:]}",
                    gross_paise=fb.gross_paise, net_paise=fb.net_paise,
                    mdr_paise=0, gst_paise=0, method="upi",
                    captured_at=when, customer_ref=f"cust_{self.rng.randint(10000, 99999)}",
                )
                payments.append(p)
                by_day.setdefault(d, []).append(p)
                group.append(p)

            target = sum(p.net_paise for p in group[:3])
            bt = BankTxn(
                bank_txn_id=f"bnk_{self._next():010d}",
                value_date=d + timedelta(days=2), amount_paise=target,
                narration=self._narration_no_utr(), utr=None,
                counterparty="RAZORPAY SOFTWARE PVT LTD",
            )
            bank_txns.append(bt)
            truth.append(GroundTruth(
                bank_txn_id=bt.bank_txn_id, scenario=Scenario.AMBIGUOUS_SUBSET,
                payment_ids=[p.payment_id for p in group[:3]], resolvable=False,
                note="Two disjoint payment sets reconcile to this amount exactly; "
                     "correct behaviour is to abstain, not to guess.",
            ))

    def _inject_orphans(self, bank_txns: list[BankTxn],
                        truth: list[GroundTruth]) -> None:
        """Credits that are not gateway settlements at all.

        A merchant's current account receives plenty of money that has nothing
        to do with the payment gateway. An engine that force-fits these into a
        settlement is inventing revenue.
        """
        n = max(2, int(len(bank_txns) * self.orphan_rate))
        for _ in range(n):
            d = self.start + timedelta(days=self.rng.randint(2, self.days + 1))
            amt = rupees(self.rng.randrange(5000, 250000, 500))
            sender = self.rng.choice(ORPHAN_SENDERS)
            bt = BankTxn(
                bank_txn_id=f"bnk_{self._next():010d}", value_date=d,
                amount_paise=amt,
                narration=f"NEFT CR-{self.rng.choice(BANKS)}0001234-{sender}-"
                          f"INV{self.rng.randint(1000, 9999)}",
                utr=None, counterparty=sender,
            )
            bank_txns.append(bt)
            truth.append(GroundTruth(
                bank_txn_id=bt.bank_txn_id, scenario=Scenario.ORPHAN_CREDIT,
                payment_ids=[], resolvable=False,
                note="Direct customer receipt, not a gateway settlement.",
            ))


def generate(seed: int = 42, n_payments: int = 600, days: int = 21,
             **kw) -> Batch:
    return Generator(seed=seed, n_payments=n_payments, days=days, **kw).build()
