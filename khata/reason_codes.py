"""Typed reasons an item lands in the exception queue.

An exception list that says "12 items failed" is worthless to the finance team
that has to clear it. Each code below carries the owner who can actually act on
it and the next physical step, because that is the difference between a report
and a workflow.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasonCode:
    code: str
    title: str
    owner: str
    next_action: str
    severity: str  # info | warn | block


_CODES: list[ReasonCode] = [
    ReasonCode(
        "NOT_A_SETTLEMENT", "Credit is not a gateway settlement",
        "Finance", "Classify as direct customer receipt and post to AR, not to gateway clearing.", "info"),
    ReasonCode(
        "AMBIGUOUS_SUBSET", "Multiple payment sets reconcile to this credit",
        "Finance", "Pull the settlement breakup report from the gateway dashboard for this UTR.", "warn"),
    ReasonCode(
        "NO_FEASIBLE_SUBSET", "No combination of open payments reaches this amount",
        "Finance", "Check for a payment captured outside the search window or an unrecorded adjustment.", "warn"),
    ReasonCode(
        "AMOUNT_SHORTFALL", "Credit exceeds every open payment still available",
        "Finance", "Likely an out-of-band credit (loan, refund reversal, prior-period sweep). Confirm with the bank.", "warn"),
    ReasonCode(
        "DUPLICATE_UTR", "This UTR was already reconciled against another credit",
        "Treasury", "Confirm with the bank whether the statement line is a duplicate posting.", "block"),
    ReasonCode(
        "LOW_CONFIDENCE", "Adjudicator declined to commit below the confidence floor",
        "Finance", "Human review: the evidence supports a match but not strongly enough to auto-post.", "warn"),
    ReasonCode(
        "NEEDS_LLM_REVIEW", "Escalated to Tier 3 but the adjudicator was unavailable",
        "Platform", "Re-run with ANTHROPIC_API_KEY set, or clear manually.", "info"),
    ReasonCode(
        "ADJUDICATOR_ERROR", "Tier 3 adjudicator failed and the credit was left untouched",
        "Platform", "Check the audit trail for the upstream error, then re-run this credit.", "warn"),
    ReasonCode(
        "SEARCH_BUDGET_EXCEEDED", "Combinatorial search hit its time budget",
        "Platform", "Narrow the date window or raise KHATA_SUBSET_BUDGET_MS, then re-run.", "info"),
    ReasonCode(
        "POOL_EXHAUSTED", "No open payments remain in the search window",
        "Finance", "Credit likely belongs to a period outside this batch. Widen the window.", "warn"),
]

BY_CODE: dict[str, ReasonCode] = {c.code: c for c in _CODES}


def get(code: str) -> ReasonCode:
    return BY_CODE.get(code, ReasonCode(
        code, code.replace("_", " ").title(), "Platform",
        "Unclassified exception -- inspect the audit trail.", "warn"))


def all_codes() -> list[ReasonCode]:
    return list(_CODES)
