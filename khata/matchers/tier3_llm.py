"""Tier 3 -- the adjudicator, for credits where only the prose is left.

By the time a credit reaches this tier the arithmetic has genuinely run out:
no UTR, no advice that nets to the amount, and no subset of open payments that
reconciles. What remains is free text a human would read -- "RAZORPAY MERCHANT
PAYOUT ORD 4471,4472,4478 AND OTHERS LESS ADJ" -- and a join no regex can
perform, because recovering the settlement from that string means mapping
order references to payments to the cycle they belonged to.

Two things keep this tier defensible:

**It runs last and rarely.** Every credit resolvable by lookup or arithmetic
has already been resolved for free. Only the residue reaches a token.

**It is allowed to abstain, and rewarded for it.** The tool contract offers
``abstain`` as a first-class decision and the prompt tells the model to prefer
it. A confident wrong attribution is far more expensive than an open item: the
open item costs an analyst five minutes, the wrong one corrupts the ledger and
is discovered at audit.

Deterministic clues are extracted before any API call, so an explicit
settlement reference or a plainly non-gateway counterparty costs nothing. With
no API key configured the tier degrades to that extraction alone, and every
credit it cannot clear is logged as NEEDS_LLM_REVIEW rather than silently
dropped.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..models import BankTxn
from .base import ESCALATE, EXCEPTION, MATCHED, Decision, MatchContext

SETTLEMENT_REF_RE = re.compile(r"\bSETL[_-]?(\d{6,12})\b", re.I)
ORDER_REF_RE = re.compile(r"\bORD[A-Z ]{0,8}((?:\d{3,12}[,/ ]{0,2}){1,8})", re.I)
GATEWAY_MARKERS = ("RAZORPAY", "RAZORPAYSO", "RZP")

MAX_ADVICES_IN_PROMPT = 25
MAX_PAYMENTS_IN_PROMPT = 40


class Verdict(BaseModel):
    """Structured adjudication. ``abstain`` is always a valid answer."""

    decision: Literal["match_settlement", "match_payments", "not_a_settlement", "abstain"]
    settlement_id: str | None = Field(
        None, description="Settlement id, required when decision is match_settlement.")
    payment_ids: list[str] = Field(
        default_factory=list,
        description="Payment ids, required when decision is match_payments.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="0-1. Below 0.8 the engine will not post the match.")
    reasoning: str = Field(
        ..., description="One or two sentences citing the specific evidence used.")
    exception_code: str | None = Field(
        None, description="On abstain: AMBIGUOUS_SUBSET, NO_FEASIBLE_SUBSET or LOW_CONFIDENCE.")


SYSTEM = """You are the final adjudicator in a settlement reconciliation engine for an \
Indian merchant on Razorpay. Deterministic tiers have already failed on this bank credit: \
there is no matching UTR, no settlement advice whose net equals the amount, and no exact \
subset of open payments that reconciles to it.

You decide only what the arithmetic could not. Weigh the narration text, the settlement \
advices offered, and any order references that were resolved for you.

Decision rules, in order of priority:

1. Prefer `abstain` whenever the evidence is merely suggestive. An unresolved credit costs \
an analyst a few minutes. A wrong attribution corrupts the ledger and surfaces at audit. \
These costs are not comparable, so do not trade one for the other.
2. Use `match_settlement` when the narration identifies a specific settlement -- by explicit \
reference, or because the order references it names all belong to one settlement cycle. A \
small unexplained difference between the credit and the advice net is normal and is usually \
a gateway adjustment; say so in your reasoning and give the amount.
3. Use `not_a_settlement` when the counterparty and narration show an unrelated inbound \
payment, such as a direct customer transfer or a vendor refund. Being confidently *not* a \
settlement is a useful answer, not a failure.
4. Use `match_payments` only when you can name the exact payment ids and their net amounts \
add up. If you are inferring which payments were included, abstain instead.
5. When two or more payment sets both explain the credit exactly, abstain with \
`AMBIGUOUS_SUBSET` unless the narration positively distinguishes them. Amount alone never \
distinguishes them -- that is what made it ambiguous.

Set `confidence` to what the evidence actually supports. The engine will not post below \
0.8, so a hedged number is how you decline without claiming the credit is inexplicable."""


class Tier3Adjudicator:
    name = "T3-ADJUDICATOR"

    def __init__(self, model: str | None = None, enabled: bool = True,
                 max_calls: int = 40):
        self.model = model or os.environ.get("KHATA_MODEL", "claude-opus-5")
        self.max_calls = max_calls
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.errors: list[str] = []
        self._client = None
        self.enabled = enabled and bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.disabled_reason = (
            "" if self.enabled else
            ("ANTHROPIC_API_KEY not set" if enabled else "disabled by --no-llm"))

    # ---- client ----

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # ---- deterministic pre-pass ----

    def _clues(self, txn: BankTxn, ctx: MatchContext) -> dict[str, Any]:
        """Extract everything recoverable without a model."""
        narr = txn.narration.upper()
        clues: dict[str, Any] = {
            "has_gateway_marker": any(m in narr for m in GATEWAY_MARKERS)
                                  or "RAZORPAY" in txn.counterparty.upper(),
            "settlement_refs": [],
            "resolved_orders": [],
        }
        for m in SETTLEMENT_REF_RE.finditer(narr):
            sid = f"setl_{int(m.group(1)):010d}"
            if sid in ctx.advice_by_id:
                clues["settlement_refs"].append(sid)

        m = ORDER_REF_RE.search(narr)
        if m:
            for tok in re.split(r"[,/ ]+", m.group(1)):
                if not tok.isdigit():
                    continue
                oid = f"order_{int(tok):012d}"
                pid = ctx.payment_by_order.get(oid)
                if not pid:
                    continue
                owner = next((a.settlement_id for a in ctx.advices
                              if a.breakup_available and pid in a.payment_ids), None)
                clues["resolved_orders"].append(
                    {"order_id": oid, "payment_id": pid,
                     "net_paise": ctx.payments[pid].net_paise,
                     "belongs_to_settlement": owner})
        return clues

    # ---- main ----

    def attempt(self, txn: BankTxn, ctx: MatchContext,
                prior: Decision | None = None) -> Decision:
        t0 = time.perf_counter()

        def done(**kw) -> Decision:
            kw.setdefault("tier", self.name)
            kw["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            return Decision(**kw)

        clues = self._clues(txn, ctx)

        # (a) An explicit, verifiable settlement reference. Unambiguous by
        # construction -- no model required, and none used.
        refs = clues["settlement_refs"]
        if len(refs) == 1:
            a = ctx.advice_by_id[refs[0]]
            if a.breakup_available and ctx.advice_remaining(a.settlement_id) > 0:
                delta = txn.amount_paise - a.net_paise
                return done(
                    outcome=MATCHED, strategy="narration_settlement_ref", confidence=0.94,
                    settlement_id=a.settlement_id, payment_ids=list(a.payment_ids),
                    residual_paise=delta,
                    evidence={"resolver": "deterministic", "settlement_ref": a.settlement_id,
                              "advice_net_paise": a.net_paise, "delta_paise": delta,
                              "payment_count": len(a.payment_ids),
                              "why": "Narration carries an explicit settlement reference that "
                                     "resolves to an advice on file; the difference is a "
                                     "post-advice gateway adjustment."})

        # (b) Nothing about this credit says "payment gateway".
        if not clues["has_gateway_marker"]:
            return done(
                outcome=EXCEPTION, reason_code="NOT_A_SETTLEMENT",
                strategy="counterparty_not_gateway", confidence=0.96,
                evidence={"resolver": "deterministic", "counterparty": txn.counterparty,
                          "narration": txn.narration,
                          "why": "Neither the counterparty nor the narration references the "
                                 "payment gateway. Classifying as a direct receipt rather "
                                 "than forcing it into a settlement."})

        fallback = (prior.fallback_reason_code if prior else None) or "NO_FEASIBLE_SUBSET"

        if not self.enabled:
            return done(
                outcome=EXCEPTION, reason_code="NEEDS_LLM_REVIEW",
                strategy="adjudicator_unavailable", confidence=0.0,
                evidence={"resolver": "none", "disabled_reason": self.disabled_reason,
                          "would_have_been": fallback,
                          "why": f"Adjudicator unavailable ({self.disabled_reason}). Credit "
                                 "left open rather than resolved by guesswork."})

        if self.calls >= self.max_calls:
            return done(
                outcome=EXCEPTION, reason_code="NEEDS_LLM_REVIEW",
                strategy="call_cap_reached", confidence=0.0,
                evidence={"resolver": "none", "max_calls": self.max_calls,
                          "why": "Per-batch adjudicator call cap reached; remaining credits "
                                 "left open rather than escalated without a budget."})

        try:
            return self._adjudicate(done, txn, ctx, clues, prior, fallback)
        except Exception as exc:                      # noqa: BLE001
            # A failing adjudicator must never take the batch down or, worse,
            # produce a match. It fails open, into the exception queue.
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return done(
                outcome=EXCEPTION, reason_code="ADJUDICATOR_ERROR",
                strategy="api_error", confidence=0.0,
                evidence={"resolver": "error", "error": f"{type(exc).__name__}: {exc}",
                          "why": "Adjudicator call failed. Credit left untouched -- a failed "
                                 "reasoning step must not become a posted match."})

    def _adjudicate(self, done, txn: BankTxn, ctx: MatchContext,
                    clues: dict, prior: Decision | None, fallback: str) -> Decision:
        payload = self._build_payload(txn, ctx, clues, prior)
        client = self._get_client()
        self.calls += 1

        resp = client.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": payload}],
            output_format=Verdict,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0

        v: Verdict = resp.parsed_output
        ev = {"resolver": "llm", "model": self.model, "decision": v.decision,
              "model_confidence": v.confidence, "reasoning": v.reasoning,
              "clues": clues, "why": v.reasoning}

        if v.decision == "not_a_settlement" and v.confidence >= ctx.confidence_floor:
            return done(outcome=EXCEPTION, reason_code="NOT_A_SETTLEMENT",
                        strategy="llm_not_a_settlement", confidence=v.confidence,
                        evidence=ev)

        if v.decision == "match_settlement" and v.settlement_id:
            a = ctx.advice_by_id.get(v.settlement_id)
            if a is None or not a.breakup_available:
                # The model named a settlement we cannot expand into payments.
                # Treat as an abstain rather than trusting an unverifiable id.
                return done(outcome=EXCEPTION, reason_code="LOW_CONFIDENCE",
                            strategy="llm_unverifiable_settlement", confidence=v.confidence,
                            evidence={**ev, "why": "Adjudicator named a settlement with no "
                                                   "payment breakup on file; cannot verify."})
            if v.confidence < ctx.confidence_floor:
                return done(outcome=EXCEPTION, reason_code="LOW_CONFIDENCE",
                            strategy="llm_below_floor", confidence=v.confidence,
                            evidence=ev)
            return done(outcome=MATCHED, strategy="llm_settlement_ref",
                        confidence=v.confidence, settlement_id=a.settlement_id,
                        payment_ids=list(a.payment_ids),
                        residual_paise=txn.amount_paise - a.net_paise,
                        evidence={**ev, "advice_net_paise": a.net_paise,
                                  "delta_paise": txn.amount_paise - a.net_paise,
                                  "payment_count": len(a.payment_ids)})

        if v.decision == "match_payments" and v.payment_ids:
            # Never take the model's word on arithmetic. Re-add the nets and
            # reject the verdict if they do not reach the credit.
            known = [p for p in v.payment_ids
                     if p in ctx.payments and p not in ctx.consumed_payments]
            total = sum(ctx.payments[p].net_paise for p in known)
            delta = total - txn.amount_paise
            ok = (len(known) == len(v.payment_ids)
                  and abs(delta) <= ctx.loose_tolerance_paise
                  and v.confidence >= ctx.confidence_floor)
            ev = {**ev, "recomputed_total_paise": total, "delta_paise": delta,
                  "unknown_or_consumed_ids": len(v.payment_ids) - len(known)}
            if ok:
                return done(outcome=MATCHED, strategy="llm_payment_set",
                            confidence=min(v.confidence, 0.90), payment_ids=known,
                            residual_paise=delta, evidence=ev)
            return done(outcome=EXCEPTION, reason_code="LOW_CONFIDENCE",
                        strategy="llm_arithmetic_rejected", confidence=v.confidence,
                        evidence={**ev, "why": "Adjudicator proposed a payment set whose nets "
                                               "do not sum to the credit. Verdict rejected."})

        code = v.exception_code if v.exception_code in ("AMBIGUOUS_SUBSET",
                                                        "NO_FEASIBLE_SUBSET",
                                                        "LOW_CONFIDENCE") else fallback
        return done(outcome=EXCEPTION, reason_code=code, strategy="llm_abstain",
                    confidence=v.confidence, evidence=ev)

    def _build_payload(self, txn: BankTxn, ctx: MatchContext,
                       clues: dict, prior: Decision | None) -> str:
        from ..money import fmt

        lines = [
            "## Bank credit to adjudicate",
            f"- id: {txn.bank_txn_id}",
            f"- value date: {txn.value_date.isoformat()}",
            f"- amount: {fmt(txn.amount_paise)} ({txn.amount_paise} paise)",
            f"- counterparty: {txn.counterparty}",
            f"- narration: {txn.narration!r}",
            f"- UTR on statement: {txn.utr or 'none'}",
            "",
            "## Why the deterministic tiers could not resolve it",
        ]
        if prior is not None:
            lines.append(f"- last tier: {prior.tier} ({prior.strategy})")
            for k, val in prior.evidence.items():
                if k != "why":
                    lines.append(f"- {k}: {val}")
            lines.append(f"- diagnosis: {prior.evidence.get('why', 'n/a')}")
        else:
            lines.append("- no prior diagnosis recorded")

        if clues["resolved_orders"]:
            lines += ["", "## Order references resolved from the narration",
                      "(order id -> payment -> the settlement cycle that payment belonged to)"]
            for r in clues["resolved_orders"]:
                lines.append(f"- {r['order_id']} -> {r['payment_id']} "
                             f"({fmt(r['net_paise'])} net) -> "
                             f"{r['belongs_to_settlement'] or 'no settlement on file'}")

        adv = ctx.advices_in_window(txn.value_date, with_breakup=True)[:MAX_ADVICES_IN_PROMPT]
        if adv:
            lines += ["", "## Open settlement advices in the search window",
                      "(payment breakup is on file for all of these)"]
            for a in adv:
                d = txn.amount_paise - a.net_paise
                lines.append(
                    f"- {a.settlement_id}: net {fmt(a.net_paise)}, "
                    f"{a.settled_at.date().isoformat()}, {len(a.payment_ids)} payments, "
                    f"credit minus advice = {fmt(d)}")

        pool = ctx.open_payments_in_window(txn.value_date)
        lines += ["", f"## Open payments in the window: {len(pool)}, "
                      f"totalling {fmt(sum(p.net_paise for p in pool))} net"]
        if len(pool) <= MAX_PAYMENTS_IN_PROMPT:
            for p in sorted(pool, key=lambda x: x.captured_at)[:MAX_PAYMENTS_IN_PROMPT]:
                lines.append(f"- {p.payment_id} ({p.order_id}) {fmt(p.net_paise)} net, "
                             f"{p.method}, {p.captured_at.date().isoformat()}")
        else:
            lines.append(f"- (list withheld: {len(pool)} payments is too many to enumerate "
                         f"reliably; do not guess a subset from summary statistics)")

        lines += ["", "Adjudicate this credit. Abstain unless the evidence is specific."]
        return "\n".join(lines)
