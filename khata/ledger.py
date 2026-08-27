"""Double-entry ledger with enforced invariants.

Reconciliation output that is not backed by a balancing ledger is a
spreadsheet with extra steps. Every event posts equal debits and credits, and
``assert_balanced`` is called after every batch -- if the engine ever invents
or destroys a rupee, the run fails loudly instead of shipping a pretty number.

Unattributed bank credits are parked in ``suspense`` rather than being forced
into the clearing account. Suspense is the honest place for money you have
received but cannot yet explain, and its closing balance is the single figure a
finance lead will look at first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# account -> normal balance side
ACCOUNTS: dict[str, str] = {
    "bank": "debit",
    "gateway_clearing": "debit",
    "mdr_expense": "debit",
    "gst_input_credit": "debit",
    "refunds_paid": "debit",
    "chargeback_losses": "debit",
    "revenue": "credit",
    "suspense": "credit",
}


@dataclass
class Posting:
    seq: int
    event: str
    ref: str
    debits: dict[str, int]
    credits: dict[str, int]

    def total_debit(self) -> int:
        return sum(self.debits.values())

    def total_credit(self) -> int:
        return sum(self.credits.values())


class LedgerImbalance(AssertionError):
    pass


@dataclass
class Ledger:
    postings: list[Posting] = field(default_factory=list)
    balances: dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACCOUNTS})

    def post(self, event: str, ref: str, debits: dict[str, int],
             credits: dict[str, int]) -> Posting:
        td, tc = sum(debits.values()), sum(credits.values())
        if td != tc:
            raise LedgerImbalance(
                f"{event}/{ref}: debits {td} != credits {tc}. Refusing to post.")
        for acct in list(debits) + list(credits):
            if acct not in ACCOUNTS:
                raise KeyError(f"Unknown account {acct!r}")
        p = Posting(len(self.postings) + 1, event, ref, dict(debits), dict(credits))
        self.postings.append(p)
        for a, v in debits.items():
            self.balances[a] += v if ACCOUNTS[a] == "debit" else -v
        for a, v in credits.items():
            self.balances[a] += v if ACCOUNTS[a] == "credit" else -v
        return p

    # ---- domain events ----

    def payment_captured(self, payment_id: str, gross: int, mdr: int,
                         gst: int, net: int) -> None:
        self.post("payment_captured", payment_id,
                  {"gateway_clearing": net, "mdr_expense": mdr, "gst_input_credit": gst},
                  {"revenue": gross})

    def refund_issued(self, refund_id: str, amount: int) -> None:
        self.post("refund_issued", refund_id,
                  {"refunds_paid": amount}, {"gateway_clearing": amount})

    def chargeback_raised(self, dispute_id: str, amount: int) -> None:
        self.post("chargeback_raised", dispute_id,
                  {"chargeback_losses": amount}, {"gateway_clearing": amount})

    def settlement_matched(self, bank_txn_id: str, amount: int) -> None:
        self.post("settlement_matched", bank_txn_id,
                  {"bank": amount}, {"gateway_clearing": amount})

    def credit_unattributed(self, bank_txn_id: str, amount: int) -> None:
        self.post("credit_unattributed", bank_txn_id,
                  {"bank": amount}, {"suspense": amount})

    def reclassify_from_suspense(self, bank_txn_id: str, amount: int) -> None:
        """Move a credit out of suspense once it has been explained.

        The cash never moves -- the bank leg stands. This is the reclassification
        journal a controller posts when an item clears the suspense account, and
        posting it (rather than editing the original entry) keeps the trail
        intact: the credit was unexplained, then it was explained, and both facts
        remain in the ledger.
        """
        self.post("suspense_cleared", bank_txn_id,
                  {"suspense": amount}, {"gateway_clearing": amount})

    # ---- invariants ----

    def trial_balance(self) -> dict[str, Any]:
        debit_total = sum(p.total_debit() for p in self.postings)
        credit_total = sum(p.total_credit() for p in self.postings)
        return {
            "postings": len(self.postings),
            "total_debits_paise": debit_total,
            "total_credits_paise": credit_total,
            "balanced": debit_total == credit_total,
            "balances": dict(self.balances),
        }

    def assert_balanced(self) -> None:
        tb = self.trial_balance()
        if not tb["balanced"]:
            raise LedgerImbalance(
                f"Trial balance broken: debits {tb['total_debits_paise']} "
                f"!= credits {tb['total_credits_paise']}")

    def suspense_balance(self) -> int:
        return self.balances["suspense"]
