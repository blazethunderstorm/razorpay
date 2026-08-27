"""Tier orchestration.

Bank credits are processed in value-date order. That is not cosmetic: each
match consumes its payments, so by the time a later credit is considered the
pool has already shrunk. Processing out of order would leave every settled
payment searchable for every subsequent credit, inflating pools into territory
where exact subset-sum stops being affordable and ambiguity becomes rampant.

Every tier attempt is written to the audit trail, including the ones that
escalate. The trail should answer "why did this credit end up here" without
anyone re-running the engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from .audit import AuditTrail
from .gateway import GatewayClient
from .ledger import Ledger
from datetime import date

from .groupsolve import solve_partition
from .matchers.tier2_subset import DAY_OFFSETS
from .subsetsum import Item, solve
from .matchers import (ESCALATE, EXCEPTION, MATCHED, Decision, MatchContext,
                       Tier0SourceCheck, Tier0UTR, Tier1Advice, Tier2Subset,
                       Tier2bCashOnly,
                       Tier3Adjudicator)
from .models import Batch, BankTxn
from .money import fmt


@dataclass
class TxnOutcome:
    bank_txn_id: str
    value_date: str
    amount_paise: int
    narration: str
    outcome: str
    tier: str
    strategy: str
    confidence: float
    reason_code: str | None
    attribution: str
    payment_ids: list[str]
    settlement_id: str | None
    residual_paise: int
    elapsed_ms: float
    evidence: dict[str, Any]
    trail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["amount_display"] = fmt(self.amount_paise)
        return d


@dataclass
class RunResult:
    batch_id: str
    outcomes: list[TxnOutcome]
    ledger: Ledger
    audit: AuditTrail
    wall_ms: float
    tier_counts: dict[str, int]
    llm: dict[str, Any]
    gateway: dict[str, Any]
    group: dict[str, Any]
    config: dict[str, Any]

    @property
    def matched(self) -> list[TxnOutcome]:
        return [o for o in self.outcomes if o.outcome == MATCHED]

    @property
    def exceptions(self) -> list[TxnOutcome]:
        return [o for o in self.outcomes if o.outcome != MATCHED]

    def by_id(self, txn_id: str) -> TxnOutcome | None:
        return next((o for o in self.outcomes if o.bank_txn_id == txn_id), None)


class Engine:
    def __init__(self, use_llm: bool = True, lookback_days: int = 8,
                 tolerance_paise: int = 0, confidence_floor: float = 0.80,
                 subset_budget_ms: int = 2000, max_llm_calls: int = 40,
                 gateway_budget: int = 40, gateway_policy: str = "fifo",
                 live_gateway: bool = False,
                 enable_group: bool = False,
                 enabled_tiers: frozenset[str] | None = None,
                 model: str | None = None, audit_path: str | None = None):
        self.t0src = Tier0SourceCheck()
        self.t0 = Tier0UTR()
        self.t1 = Tier1Advice()
        self.t2 = Tier2Subset()
        self.t2b = Tier2bCashOnly()
        self.t3 = Tier3Adjudicator(model=model, enabled=use_llm, max_calls=max_llm_calls)
        self.lookback_days = lookback_days
        self.tolerance_paise = tolerance_paise
        self.confidence_floor = confidence_floor
        self.subset_budget_ms = subset_budget_ms
        self.gateway_budget = gateway_budget
        # "fifo"  -- spend calls on whichever credit reaches Tier 2b first.
        # "value" -- hold every request, then spend the budget largest-credit
        #            first once the whole batch is known.
        #
        # "value" is the obvious optimisation and it loses. Measured over six
        # seeds at budgets 4-40 it is behind fifo on both line recall and on
        # rupees explained, by up to 8.6 points. The reason is that a recon
        # call's worth is not the size of the credit that triggered it: the
        # breakup is written back onto the advice, so an early call also feeds
        # every later credit and tier. Deferring the call to aim it better
        # destroys exactly that compounding, and the aim is worth less than the
        # compounding. Kept, defaulted off, because the measurement is the
        # useful part -- see README.
        if gateway_policy not in ("fifo", "value"):
            raise ValueError(f"gateway_policy must be 'fifo' or 'value', got {gateway_policy!r}")
        self.gateway_policy = gateway_policy
        # Swap the simulated settlement endpoints for the real ones. The tiers
        # are unchanged: the client honours the same budget and the same
        # "identify the settlement yourself first" contract.
        self.live_gateway = live_gateway
        # The exact-cover group pass is off by default. It is correct, but on
        # realistic pool sizes it has never once proved a unique partition, and
        # it is not free -- see the ablation in the README.
        self.enable_group = enable_group
        self.enabled_tiers = enabled_tiers
        self.audit_path = audit_path

    def run(self, batch: Batch) -> RunResult:
        start = time.perf_counter()
        ledger = Ledger()
        audit = AuditTrail(self.audit_path)

        # Opening position: everything the merchant's own books already know,
        # before a single bank line is looked at.
        for p in batch.payments:
            ledger.payment_captured(p.payment_id, p.gross_paise, p.mdr_paise,
                                    p.gst_paise, p.net_paise)
        for r in batch.refunds:
            ledger.refund_issued(r.refund_id, r.amount_paise)
        for c in batch.chargebacks:
            ledger.chargeback_raised(c.dispute_id, c.amount_paise)
        ledger.assert_balanced()

        if self.live_gateway:
            from .razorpay_client import RazorpaySettlements
            gateway: GatewayClient = RazorpaySettlements(
                ledger=list(batch.gateway_ledger), budget=self.gateway_budget)
        else:
            gateway = GatewayClient(ledger=list(batch.gateway_ledger),
                                    budget=self.gateway_budget)
        # Under "value", Tier 2b's recon requests are held rather than served,
        # then replayed largest credit first once the whole batch is known.
        gateway.hold = self.gateway_policy == "value"
        # Copy the merchant's advices before handing them to the matchers.
        # Recovering a breakup from the gateway writes it back onto the advice so
        # a second statement leg reuses it instead of paying twice -- which is
        # right within a run and wrong across runs. Mutating the batch made every
        # repeat run look 55% cheaper than the first, because the breakups it had
        # already bought were still sitting there. Reconciling the same batch
        # twice must cost the same twice.
        advices = [replace(a, payment_ids=list(a.payment_ids)) for a in batch.advices]
        ctx = MatchContext(
            gateway=gateway,
            payments={p.payment_id: p for p in batch.payments},
            refunds={r.refund_id: r for r in batch.refunds},
            chargebacks={c.dispute_id: c for c in batch.chargebacks},
            advices=advices,
            bank_txns=list(batch.bank_txns),
            lookback_days=self.lookback_days,
            tolerance_paise=self.tolerance_paise,
            subset_budget_ms=self.subset_budget_ms,
            confidence_floor=self.confidence_floor,
        )

        tier_counts: dict[str, int] = {}
        outcomes: list[TxnOutcome] = []

        for txn in sorted(batch.bank_txns, key=lambda t: (t.value_date, t.bank_txn_id)):
            t_start = time.perf_counter()
            final, trail = self._resolve(txn, ctx)
            wall = round((time.perf_counter() - t_start) * 1000, 3)

            if final.outcome == MATCHED:
                ctx.consume(final, txn)
                ledger.settlement_matched(txn.bank_txn_id, txn.amount_paise)
                key = f"{final.tier}"
            else:
                # Unattributed cash goes to suspense, never to clearing. The
                # closing suspense balance is then exactly the money we have
                # received and cannot yet explain.
                ledger.credit_unattributed(txn.bank_txn_id, txn.amount_paise)
                key = f"{final.tier}:exception"
            tier_counts[key] = tier_counts.get(key, 0) + 1

            for d in trail:
                audit.log(bank_txn_id=txn.bank_txn_id, amount_paise=txn.amount_paise,
                          tier=d.tier, outcome=d.outcome, reason_code=d.reason_code,
                          confidence=d.confidence, payment_ids=d.payment_ids,
                          settlement_id=d.settlement_id, residual_paise=d.residual_paise,
                          strategy=d.strategy, elapsed_ms=d.elapsed_ms, evidence=d.evidence)

            outcomes.append(TxnOutcome(
                bank_txn_id=txn.bank_txn_id, value_date=txn.value_date.isoformat(),
                amount_paise=txn.amount_paise, narration=txn.narration,
                outcome=final.outcome, tier=final.tier, strategy=final.strategy,
                confidence=final.confidence, reason_code=final.reason_code,
                attribution=final.attribution if final.outcome == MATCHED else "none",
                payment_ids=list(final.payment_ids), settlement_id=final.settlement_id,
                residual_paise=final.residual_paise, elapsed_ms=wall,
                evidence=final.evidence,
                trail=[{"tier": d.tier, "outcome": d.outcome, "strategy": d.strategy,
                        "reason_code": d.reason_code, "confidence": d.confidence,
                        "elapsed_ms": d.elapsed_ms,
                        "why": d.evidence.get("why", "")} for d in trail],
            ))

        # Phase 1b -- spend the held gateway budget on the largest credits.
        gateway_upgrades = self._gateway_pass(ctx, outcomes, audit, tier_counts,
                                              {t.bank_txn_id: t for t in batch.bank_txns})

        # Phase 2 -- reconcile each day's leftovers jointly.
        group_upgrades = (self._group_pass(ctx, outcomes, ledger, audit, tier_counts)
                          if self.enable_group else
                          {"enabled": False, "groups_considered": 0, "groups_solved": 0,
                           "credits_upgraded": 0, "cash_to_line": 0, "reopened": 0,
                           "ambiguous_groups": 0, "amount_upgraded_paise": 0,
                           "details": []})

        # If this ever fires the run is void, however good the match rate looks.
        ledger.assert_balanced()

        return RunResult(
            batch_id=batch.batch_id, outcomes=outcomes, ledger=ledger, audit=audit,
            wall_ms=round((time.perf_counter() - start) * 1000, 2),
            tier_counts=tier_counts,
            llm={"enabled": self.t3.enabled, "disabled_reason": self.t3.disabled_reason,
                 "calls": self.t3.calls, "input_tokens": self.t3.input_tokens,
                 "output_tokens": self.t3.output_tokens, "model": self.t3.model,
                 "errors": self.t3.errors},
            gateway={**gateway.summary(), "value_pass": gateway_upgrades},
            group=group_upgrades,
            config={"lookback_days": self.lookback_days,
                    "gateway_budget": self.gateway_budget,
                    "gateway_policy": self.gateway_policy,
                    "gateway_live": self.live_gateway,
                    "group_pass": self.enable_group,
                    "enabled_tiers": sorted(self.enabled_tiers) if self.enabled_tiers else "all",
                    "tolerance_paise": self.tolerance_paise,
                    "confidence_floor": self.confidence_floor,
                    "subset_budget_ms": self.subset_budget_ms},
        )

    # Exception codes worth a second look in the group pass. Deliberately
    # excludes the codes where abstaining was a positive classification --
    # re-opening an orphan receipt or a duplicate bank posting risks turning a
    # correct abstention into a false match, which is the one trade this engine
    # never makes.
    GROUP_RETRY_CODES = frozenset({
        "AMBIGUOUS_SUBSET", "NO_FEASIBLE_SUBSET", "NEEDS_LLM_REVIEW",
        "LOW_CONFIDENCE", "POOL_EXHAUSTED", "SEARCH_BUDGET_EXCEEDED",
        "AMOUNT_SHORTFALL", "ADJUDICATOR_ERROR",
    })

    def _gateway_pass(self, ctx: MatchContext, outcomes: list[TxnOutcome],
                      audit: AuditTrail, tier_counts: dict[str, int],
                      txn_by_id: dict[str, Any]) -> dict[str, Any]:
        """Spend the held gateway budget on the largest credits first.

        Tier 2b decides *which settlement* a credit belongs to; the recon
        endpoint only fills in *which payments*. That means the fetch can be
        postponed without changing any attribution -- the settlement is already
        named -- so the whole batch's requests are collected first and the
        budget is then aimed at the biggest unexplained amounts.

        Every upgrade here is cash_only -> line_level on a credit that was
        already matched, so no cash moves and the ledger is untouched. A
        recovered breakup that does not reconcile is discarded exactly as it is
        in Tier 2b: an unverified breakup is not evidence.
        """
        gw = ctx.gateway
        stats: dict[str, Any] = {
            "enabled": gw.hold, "held": len(gw.held), "served": 0,
            "credits_upgraded": 0, "amount_upgraded_paise": 0,
            "rejected_not_reconciling": 0, "unserved_no_budget": 0,
        }
        if not gw.hold:
            return stats
        gw.hold = False  # the holding period is over; calls are served for real

        by_id = {o.bank_txn_id: o for o in outcomes}
        ranked = sorted(gw.held, reverse=True,
                        key=lambda h: (txn_by_id[h.on_behalf_of].amount_paise,
                                       h.on_behalf_of))
        for h in ranked:
            o = by_id.get(h.on_behalf_of)
            if o is None or o.attribution != "cash_only":
                continue  # something else already explained it
            if gw.remaining <= 0:
                stats["unserved_no_budget"] += 1
                continue
            txn = txn_by_id[h.on_behalf_of]
            pids = gw.fetch_recon(h.settlement_id, h.on_behalf_of)
            stats["served"] += 1
            if not pids:
                continue
            fresh = [p for p in pids if p not in ctx.consumed_payments]
            total = sum(ctx.payments[p].net_paise for p in fresh if p in ctx.payments)
            netting = sum(c.amount_paise
                          for c in ctx.open_credits_in_window(txn.value_date))
            reconciles = (abs(total - txn.amount_paise) <= ctx.tolerance_paise
                          or 0 <= total - txn.amount_paise <= netting)
            if not (fresh and reconciles):
                stats["rejected_not_reconciling"] += 1
                continue

            ctx.mark_breakup_recovered(h.settlement_id, pids)
            o.attribution = "line_level"
            o.payment_ids = fresh
            o.strategy = "gateway_recon[value pass]"
            o.confidence = 0.96
            o.residual_paise = total - txn.amount_paise
            o.evidence = {**o.evidence, "gateway_recon": "hit (value pass)",
                          "recon_payments": len(pids), "recon_unconsumed": len(fresh),
                          "recon_net_paise": total,
                          "recon_delta_paise": total - txn.amount_paise,
                          "upgraded_from": "cash_only",
                          "why": "Settlement identified by Tier 2b; the breakup fetch was "
                                 "held until the whole batch was known, then spent here "
                                 "because this was among the largest credits still "
                                 "missing its line detail."}
            for pid in fresh:
                ctx.consumed_payments.add(pid)
            stats["credits_upgraded"] += 1
            stats["amount_upgraded_paise"] += txn.amount_paise
            audit.log(bank_txn_id=o.bank_txn_id, amount_paise=o.amount_paise,
                      tier=o.tier, outcome=MATCHED, reason_code=None,
                      confidence=o.confidence, payment_ids=fresh,
                      settlement_id=o.settlement_id, residual_paise=o.residual_paise,
                      strategy=o.strategy, elapsed_ms=0.0, evidence=o.evidence)
        return stats

    def _group_pass(self, ctx: MatchContext, outcomes: list[TxnOutcome],
                    ledger: Ledger, audit: AuditTrail,
                    tier_counts: dict[str, int]) -> dict[str, Any]:
        """Second phase: solve each day's unresolved credits as one partition.

        Credit-by-credit matching treats a day's settlement cycles as
        independent when they are not -- they divide one pool of payments and no
        payment belongs to two of them. Adding that constraint resolves
        ambiguity that is genuinely unresolvable one credit at a time.
        """
        stats = {"groups_considered": 0, "groups_solved": 0, "credits_upgraded": 0,
                 "cash_to_line": 0, "reopened": 0, "ambiguous_groups": 0,
                 "amount_upgraded_paise": 0, "details": []}

        pending: dict[str, list[TxnOutcome]] = {}
        for o in outcomes:
            needs = (o.outcome == MATCHED and o.attribution == "cash_only") or (
                o.outcome == EXCEPTION and not o.payment_ids
                and o.reason_code in self.GROUP_RETRY_CODES)
            if needs:
                pending.setdefault(o.value_date, []).append(o)

        for value_date, group in sorted(pending.items()):
            if len(group) < 2:
                continue
            stats["groups_considered"] += 1
            vdate = date.fromisoformat(value_date)
            solved = False

            # Refunds and chargebacks raised in the cycle net off one of these
            # payouts, so they belong in the pool as negative items -- a
            # partition of payments alone cannot explain a netted credit.
            netting = [
                Item(getattr(c, "refund_id", None) or c.dispute_id, -c.amount_paise)
                for c in ctx.open_credits_in_window(vdate)
            ]

            for offset in DAY_OFFSETS:
                if offset > ctx.lookback_days:
                    continue
                pool = ctx.open_payments_between(vdate, offset, offset)
                if len(pool) < 2:
                    continue
                items = [Item(p.payment_id, p.net_paise) for p in pool] + netting

                # Screen out credits this day cannot explain at all. One
                # unexplainable member -- a payout carrying an adjustment we
                # cannot see, or a credit belonging to another day entirely --
                # would otherwise make the whole partition infeasible and take
                # its solvable siblings down with it.
                feasible, dropped = [], []
                for o in group:
                    probe = solve(items, o.amount_paise, tolerance=0, budget_ms=250)
                    (feasible if probe.solutions else dropped).append(o)
                if len(feasible) < 2:
                    continue

                targets = [o.amount_paise for o in feasible]
                res = solve_partition(items, targets, budget_ms=self.subset_budget_ms)
                if res.ambiguous:
                    stats["ambiguous_groups"] += 1
                    stats["details"].append({
                        "value_date": value_date,
                        "credits": [o.bank_txn_id for o in feasible],
                        "capture_day": f"T+{offset}", "result": "ambiguous",
                        "note": res.note})
                    break
                if not res.unique:
                    continue

                a = res.assignments[0]
                stats["groups_solved"] += 1
                for idx, sol in sorted(a.picks):
                    o = feasible[idx]
                    pids = [k for k in sol.keys if k.startswith("pay_")]
                    was = o.outcome
                    if was == EXCEPTION:
                        ledger.reclassify_from_suspense(o.bank_txn_id, o.amount_paise)
                        stats["reopened"] += 1
                    else:
                        stats["cash_to_line"] += 1
                    o.outcome = MATCHED
                    o.attribution = "line_level"
                    o.tier = "T4-GROUP"
                    o.strategy = f"partition[T+{offset} capture day]"
                    o.confidence = 0.92
                    o.reason_code = None
                    o.payment_ids = pids
                    o.residual_paise = sol.residual
                    o.evidence = {
                        "phase": "group reconciliation",
                        "value_date": value_date, "capture_day": f"T+{offset}",
                        "credits_in_group": [g.bank_txn_id for g in feasible],
                        "credits_dropped_as_unexplainable": [g.bank_txn_id for g in dropped],
                        "group_targets_paise": targets,
                        "pool_size": len(pool), "explored": res.explored,
                        "partition_ms": res.elapsed_ms,
                        "search_complete": res.complete,
                        "payment_count": len(pids),
                        "upgraded_from": "cash_only" if was == MATCHED else "exception",
                        "why": (f"{len(feasible)} credits settling on {value_date} were solved "
                                f"as one partition of the T+{offset} capture day. Exactly one "
                                "assignment of disjoint payment sets explains all of them, "
                                "which the per-credit search could not establish alone."),
                    }
                    for pid in pids:
                        ctx.consumed_payments.add(pid)
                    ctx.matched_txns.add(o.bank_txn_id)
                    stats["credits_upgraded"] += 1
                    stats["amount_upgraded_paise"] += o.amount_paise
                    tier_counts["T4-GROUP"] = tier_counts.get("T4-GROUP", 0) + 1
                    audit.log(bank_txn_id=o.bank_txn_id, amount_paise=o.amount_paise,
                              tier="T4-GROUP", outcome=MATCHED, reason_code=None,
                              confidence=o.confidence, payment_ids=pids,
                              settlement_id=o.settlement_id,
                              residual_paise=o.residual_paise,
                              strategy=o.strategy, elapsed_ms=res.elapsed_ms,
                              evidence=o.evidence)
                    o.trail.append({"tier": "T4-GROUP", "outcome": MATCHED,
                                    "strategy": o.strategy, "reason_code": None,
                                    "confidence": o.confidence,
                                    "elapsed_ms": res.elapsed_ms,
                                    "why": o.evidence["why"]})
                stats["details"].append({
                    "value_date": value_date,
                    "credits": [o.bank_txn_id for o in feasible],
                    "dropped": [o.bank_txn_id for o in dropped],
                    "capture_day": f"T+{offset}", "result": "solved",
                    "pool_size": len(pool)})
                solved = True
                break

            if not solved and stats["details"][-1:] and \
                    stats["details"][-1].get("value_date") != value_date:
                stats["details"].append({"value_date": value_date,
                                         "credits": [o.bank_txn_id for o in group],
                                         "result": "no unique partition"})
        return stats

    def _resolve(self, txn: BankTxn, ctx: MatchContext) -> tuple[Decision, list[Decision]]:
        """Walk the tiers until one commits or all of them decline."""
        trail: list[Decision] = []
        prior: Decision | None = None

        for tier in (self.t0src, self.t0, self.t1, self.t2, self.t2b):
            if self.enabled_tiers is not None and tier.name not in self.enabled_tiers:
                continue
            d = tier.attempt(txn, ctx)
            trail.append(d)
            if d.outcome in (MATCHED, EXCEPTION):
                return d, trail
            prior = d

        if self.enabled_tiers is not None and self.t3.name not in self.enabled_tiers:
            d = prior or Decision(
                tier="ENGINE", outcome=EXCEPTION, reason_code="NO_FEASIBLE_SUBSET",
                strategy="tier_disabled",
                evidence={"why": "No enabled tier could resolve this credit."})
            if d.outcome == ESCALATE:
                d = Decision(
                    tier=d.tier, outcome=EXCEPTION,
                    reason_code=d.fallback_reason_code or "NO_FEASIBLE_SUBSET",
                    strategy=d.strategy, evidence=d.evidence)
            trail.append(d)
            return d, trail

        d = self.t3.attempt(txn, ctx, prior)
        trail.append(d)
        if d.outcome == ESCALATE:
            # Nothing above the adjudicator. An escalation with nowhere to go
            # is an exception, and is recorded as one rather than vanishing.
            d.outcome = EXCEPTION
            d.reason_code = d.reason_code or (prior.fallback_reason_code if prior else None) \
                or "NO_FEASIBLE_SUBSET"
        return d, trail
