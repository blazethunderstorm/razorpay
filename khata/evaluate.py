"""Scoring against held-out ground truth.

Definitions, stated up front because a match rate means nothing without them:

* A credit is **resolvable** if a correct payment attribution exists. Orphan
  receipts, duplicate bank postings and genuinely ambiguous credits are *not*
  resolvable -- the correct action on those is to abstain, and abstaining is
  scored as a win, not a miss.
* Results are scored at **two levels**, because they answer different
  questions and averaging them would hide which one failed.
  **Cash-level** asks "is this bank credit attributed to the right settlement" --
  the question that closes the books. **Line-level** asks "do we know exactly
  which payments it comprises" -- the question that ages receivables by order.
  Line-level is strictly harder and is only attempted when the payment breakup
  is recoverable at all.
* **Correct** at line level means the predicted payment-id set equals the true
  set exactly. Partial credit is not given. Getting 22 of 23 payments right
  still misstates which orders were paid for.
* One exception, and it is a real one rather than a convenience. Zero-MDR UPI
  means two payments of the same rupee value settle for the same net, so a set
  that swaps one for the other has an identical amount multiset and an
  identical total. No evidence anywhere in the data distinguishes them. Those
  are scored as **line_equivalent**: cash-correct, not strictly line-correct,
  and counted separately so the number is visible rather than buried in either
  the wins or the losses.
* **Wrong** is any posted match that is not exactly correct, *including*
  matching something that should have been left open. This is the number that
  costs real money, so it is tracked in rupees as well as in counts.
* **Precision** is over posted matches only, because that is the population a
  finance team actually inherits. **Recall** is over resolvable credits.

The two headline numbers are deliberately kept apart. An engine can reach 100%
recall by matching everything, and 100% precision by matching nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engine import RunResult
from .models import Batch
from .money import fmt, pct


@dataclass
class CreditVerdict:
    bank_txn_id: str
    scenario: str
    amount_paise: int
    resolvable: bool
    posted: bool
    attribution: str
    cash_correct: bool
    line_correct: bool
    line_equivalent: bool
    tier: str
    reason_code: str | None
    confidence: float
    predicted: list[str]
    expected: list[str]
    note: str = ""

    @property
    def label(self) -> str:
        if self.posted and self.line_correct:
            return "line_matched"
        if self.posted and self.line_equivalent:
            return "line_equivalent"
        if self.posted and self.cash_correct:
            return "cash_matched"
        if self.posted:
            return "false_match"
        if self.resolvable:
            return "missed"
        return "correct_abstention"


@dataclass
class Metrics:
    total_credits: int
    resolvable: int
    posted: int
    cash_correct: int
    line_correct: int
    line_equivalent: int
    cash_only: int
    false_matches: int
    missed: int
    correct_abstentions: int

    cash_precision: float
    cash_recall: float
    line_precision: float
    line_recall: float
    line_recall_effective: float
    f1: float
    decision_accuracy: float
    abstention_accuracy: float

    amount_total_paise: int
    amount_resolvable_paise: int
    amount_cash_correct_paise: int
    amount_line_correct_paise: int
    amount_false_matched_paise: int
    amount_suspense_paise: int

    throughput_credits_per_s: float
    wall_ms: float
    llm_calls: int
    llm_tokens: int
    payments: int

    per_scenario: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_tier: dict[str, dict[str, Any]] = field(default_factory=dict)
    exceptions: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[CreditVerdict] = field(default_factory=list)
    ledger_balanced: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "verdicts"}
        d["verdicts"] = [v.__dict__ | {"label": v.label} for v in self.verdicts]
        return d


def evaluate(batch: Batch, result: RunResult) -> Metrics:
    truth = {g.bank_txn_id: g for g in batch.ground_truth}
    pay_nets = {p.payment_id: p.net_paise for p in batch.payments}
    verdicts: list[CreditVerdict] = []

    for o in result.outcomes:
        g = truth[o.bank_txn_id]
        expected = set(g.payment_ids) if g.resolvable else set()
        predicted = set(o.payment_ids)
        posted = o.outcome == "matched"

        line_correct = (posted and g.resolvable and o.attribution == "line_level"
                        and predicted == expected)
        # Same amount multiset, different ids: an indistinguishable relabelling
        # of the same cash, not a wrong answer.
        line_equivalent = bool(
            posted and g.resolvable and o.attribution == "line_level"
            and not line_correct and predicted and len(predicted) == len(expected)
            and sorted(pay_nets[p] for p in predicted if p in pay_nets)
            == sorted(pay_nets[p] for p in expected if p in pay_nets))
        # Cash is only credited as correct when it is *verifiable*: either the
        # payment set is provably right, or we named the settlement and it was
        # the right one. A line-level match with the wrong payment set is
        # counted as a false match even if the amount happened to be right --
        # we cannot prove which settlement it belonged to, and an unprovable
        # match is not a match.
        cash_correct = posted and g.resolvable and (
            line_correct or line_equivalent
            or (o.settlement_id is not None and o.settlement_id == g.settlement_id))

        verdicts.append(CreditVerdict(
            bank_txn_id=o.bank_txn_id, scenario=g.scenario.value,
            amount_paise=o.amount_paise, resolvable=g.resolvable, posted=posted,
            attribution=o.attribution, cash_correct=cash_correct,
            line_correct=line_correct, line_equivalent=line_equivalent,
            tier=o.tier, reason_code=o.reason_code,
            confidence=o.confidence, predicted=sorted(predicted),
            expected=sorted(expected), note=g.note,
        ))

    posted = sum(v.posted for v in verdicts)
    cash_correct = sum(v.cash_correct for v in verdicts)
    line_correct = sum(v.line_correct for v in verdicts)
    line_equivalent = sum(v.line_equivalent for v in verdicts)
    cash_only = sum(1 for v in verdicts if v.label == "cash_matched")
    false_matches = posted - cash_correct
    resolvable = sum(v.resolvable for v in verdicts)
    missed = sum(1 for v in verdicts if v.label == "missed")
    correct_abstentions = sum(1 for v in verdicts if v.label == "correct_abstention")
    non_resolvable = len(verdicts) - resolvable
    posted_line = sum(1 for v in verdicts if v.posted and v.attribution == "line_level")

    cash_precision = round(cash_correct / posted, 4) if posted else 0.0
    cash_recall = round(cash_correct / resolvable, 4) if resolvable else 0.0
    line_precision = round(line_correct / posted_line, 4) if posted_line else 0.0
    line_recall = round(line_correct / resolvable, 4) if resolvable else 0.0
    line_recall_effective = round((line_correct + line_equivalent) / resolvable, 4) \
        if resolvable else 0.0
    f1 = round(2 * cash_precision * cash_recall / (cash_precision + cash_recall), 4) \
        if (cash_precision + cash_recall) else 0.0

    amt_total = sum(v.amount_paise for v in verdicts)
    amt_resolvable = sum(v.amount_paise for v in verdicts if v.resolvable)
    amt_cash = sum(v.amount_paise for v in verdicts if v.cash_correct)
    amt_line = sum(v.amount_paise for v in verdicts if v.line_correct)
    amt_false = sum(v.amount_paise for v in verdicts if v.label == "false_match")

    per_scenario: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        s = per_scenario.setdefault(v.scenario, {
            "total": 0, "resolvable": 0, "line_matched": 0, "line_equivalent": 0,
            "cash_matched": 0, "false_match": 0, "missed": 0,
            "correct_abstention": 0, "amount_paise": 0})
        s["total"] += 1
        s["resolvable"] += int(v.resolvable)
        s["amount_paise"] += v.amount_paise
        s[v.label] += 1
    for s in per_scenario.values():
        s["cash_recall_pct"] = pct(
            s["line_matched"] + s["line_equivalent"] + s["cash_matched"], s["resolvable"])
        s["line_recall_pct"] = pct(s["line_matched"], s["resolvable"])
        s["amount_display"] = fmt(s["amount_paise"])

    per_tier: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        t = per_tier.setdefault(v.tier, {"posted": 0, "cash_correct": 0,
                                         "line_correct": 0, "false_matches": 0,
                                         "exceptions": 0, "amount_paise": 0})
        if v.posted:
            t["posted"] += 1
            t["amount_paise"] += v.amount_paise
            t["cash_correct"] += int(v.cash_correct)
            t["line_correct"] += int(v.line_correct)
            t["false_matches"] += int(not v.cash_correct)
        else:
            t["exceptions"] += 1
    for t in per_tier.values():
        t["precision_pct"] = pct(t["cash_correct"], t["posted"])
        t["amount_display"] = fmt(t["amount_paise"])

    exc: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        if v.posted:
            continue
        code = v.reason_code or "UNCLASSIFIED"
        e = exc.setdefault(code, {"code": code, "count": 0, "amount_paise": 0,
                                  "correct_to_abstain": 0, "examples": []})
        e["count"] += 1
        e["amount_paise"] += v.amount_paise
        e["correct_to_abstain"] += int(not v.resolvable)
        if len(e["examples"]) < 4:
            e["examples"].append(v.bank_txn_id)
    for e in exc.values():
        e["amount_display"] = fmt(e["amount_paise"])

    return Metrics(
        total_credits=len(verdicts), resolvable=resolvable, posted=posted,
        cash_correct=cash_correct, line_correct=line_correct,
        line_equivalent=line_equivalent, cash_only=cash_only,
        false_matches=false_matches, missed=missed,
        correct_abstentions=correct_abstentions,
        cash_precision=cash_precision, cash_recall=cash_recall,
        line_precision=line_precision, line_recall=line_recall,
        line_recall_effective=line_recall_effective, f1=f1,
        decision_accuracy=round((cash_correct + correct_abstentions) / len(verdicts), 4)
        if verdicts else 0.0,
        abstention_accuracy=round(correct_abstentions / non_resolvable, 4)
        if non_resolvable else 1.0,
        amount_total_paise=amt_total, amount_resolvable_paise=amt_resolvable,
        amount_cash_correct_paise=amt_cash, amount_line_correct_paise=amt_line,
        amount_false_matched_paise=amt_false,
        amount_suspense_paise=result.ledger.suspense_balance(),
        throughput_credits_per_s=round(len(verdicts) / (result.wall_ms / 1000), 1)
        if result.wall_ms else 0.0,
        wall_ms=result.wall_ms, llm_calls=result.llm["calls"],
        llm_tokens=result.llm["input_tokens"] + result.llm["output_tokens"],
        payments=len(batch.payments),
        per_scenario=dict(sorted(per_scenario.items())),
        per_tier=dict(sorted(per_tier.items())),
        exceptions=sorted(exc.values(), key=lambda e: -e["count"]),
        verdicts=verdicts,
        ledger_balanced=result.ledger.trial_balance()["balanced"],
    )
