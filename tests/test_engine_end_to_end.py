"""End-to-end guarantees.

Run offline (no adjudicator) so the suite is deterministic, free, and does not
require an API key. The properties asserted here are the ones a reviewer should
be able to check without trusting any headline number.
"""
import pytest

from khata.engine import Engine
from khata.evaluate import evaluate
from khata.generator import generate
from khata.models import Scenario

SEEDS = [42, 143, 244, 345, 446, 547]


@pytest.fixture(scope="module")
def runs():
    out = []
    for seed in SEEDS:
        batch = generate(seed=seed)
        result = Engine(use_llm=False).run(batch)
        out.append((batch, result, evaluate(batch, result)))
    return out


def test_ground_truth_is_never_handed_to_the_engine(runs):
    batch, _, _ = runs[0]
    assert "ground_truth" not in batch.visible()
    # Settlements without a breakup must expose an empty payment list.
    for a in batch.advices:
        if not a.breakup_available:
            assert a.to_dict()["payment_ids"] == []
            assert a.payment_ids, "ground truth still holds the real set"


def test_every_break_class_is_exercised(runs):
    batch, _, _ = runs[0]
    seen = {g.scenario for g in batch.ground_truth}
    assert seen == set(Scenario), f"missing: {set(Scenario) - seen}"


def test_ledger_balances_on_every_seed(runs):
    for _, result, m in runs:
        assert result.ledger.trial_balance()["balanced"]
        assert m.ledger_balanced


def test_no_false_matches_on_any_seed(runs):
    """The headline claim. A false match is worse than an open item, so this is
    the one assertion allowed to fail the build."""
    for batch, _, m in runs:
        assert m.false_matches == 0, (
            f"seed {batch.seed}: {m.false_matches} false matches, "
            f"{m.amount_false_matched_paise} paise wrongly attributed")
        assert m.amount_false_matched_paise == 0


def test_cash_recall_holds_above_ninety_percent(runs):
    for batch, _, m in runs:
        assert m.cash_recall >= 0.90, f"seed {batch.seed}: {m.cash_recall}"


def test_no_payment_is_attributed_to_two_credits(runs):
    """Double-counting a payment overstates settled revenue."""
    for batch, result, _ in runs:
        seen: dict[str, str] = {}
        for o in result.outcomes:
            if o.outcome != "matched" or o.settlement_id:
                continue          # advice-backed legs legitimately share a set
            for pid in o.payment_ids:
                assert pid not in seen, f"{pid} claimed by {seen[pid]} and {o.bank_txn_id}"
                seen[pid] = o.bank_txn_id


def test_orphan_credits_are_never_matched(runs):
    for batch, _, m in runs:
        for v in m.verdicts:
            if v.scenario == Scenario.ORPHAN_CREDIT.value:
                assert not v.posted, f"{v.bank_txn_id}: invented a settlement"


def test_duplicate_postings_are_never_double_matched(runs):
    for batch, _, m in runs:
        dupes = [v for v in m.verdicts
                 if v.scenario == Scenario.DUPLICATE_UTR.value and not v.resolvable]
        for v in dupes:
            assert not v.posted, "reconciled the same UTR twice"


def test_planted_ambiguities_are_abstained_on(runs):
    for batch, _, m in runs:
        for v in m.verdicts:
            if v.scenario == Scenario.AMBIGUOUS_SUBSET.value:
                assert not v.posted, "guessed between indistinguishable payment sets"


def test_every_credit_reaches_a_recorded_decision(runs):
    for batch, result, _ in runs:
        assert len(result.outcomes) == len(batch.bank_txns)
        for o in result.outcomes:
            assert o.outcome in ("matched", "exception")
            assert o.trail, "no audit trail recorded"
            if o.outcome == "exception":
                assert o.reason_code, "untyped exception"


def test_suspense_equals_the_unexplained_cash(runs):
    for _, result, m in runs:
        unmatched = sum(o.amount_paise for o in result.outcomes
                        if o.outcome != "matched")
        assert result.ledger.suspense_balance() == unmatched


def test_adjudicator_off_means_zero_tokens(runs):
    for _, result, _ in runs:
        assert result.llm["calls"] == 0
        assert result.llm["input_tokens"] == 0
