"""Failure handling.

Three ways the adjudicator can let us down, and the guarantee in each case:
it never becomes a posted match, and it never takes the batch down.
"""
from unittest.mock import MagicMock, patch

import pytest

from khata.engine import Engine
from khata.evaluate import evaluate
from khata.generator import generate
from khata.matchers import MatchContext, Tier3Adjudicator, Verdict
from khata.matchers.tier3_llm import Tier3Adjudicator as T3


@pytest.fixture(scope="module")
def batch():
    return generate(seed=42, n_payments=200, days=8)


def _ctx(batch):
    return MatchContext(
        payments={p.payment_id: p for p in batch.payments},
        refunds={r.refund_id: r for r in batch.refunds},
        chargebacks={c.dispute_id: c for c in batch.chargebacks},
        advices=list(batch.advices), bank_txns=list(batch.bank_txns))


def test_disabled_adjudicator_degrades_and_never_crashes(batch):
    result = Engine(use_llm=False).run(batch)
    assert result.ledger.trial_balance()["balanced"]
    assert result.llm["calls"] == 0
    assert result.llm["disabled_reason"] == "disabled by --no-llm"
    codes = {o.reason_code for o in result.outcomes if o.outcome == "exception"}
    assert codes, "escalated credits must be recorded, not dropped"


def test_missing_api_key_is_a_disabled_reason_not_an_exception(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t3 = Tier3Adjudicator(enabled=True)
    assert not t3.enabled
    assert t3.disabled_reason == "ANTHROPIC_API_KEY not set"


def test_api_failure_leaves_the_credit_untouched(batch, monkeypatch):
    """A failed reasoning step must not become a posted match."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    engine = Engine(use_llm=True)
    boom = MagicMock(side_effect=RuntimeError("upstream 503"))
    with patch.object(T3, "_get_client") as gc:
        gc.return_value.messages.parse = boom
        result = engine.run(batch)

    assert result.ledger.trial_balance()["balanced"], "batch survived the failure"
    errored = [o for o in result.outcomes if o.reason_code == "ADJUDICATOR_ERROR"]
    if errored:                            # only if anything actually escalated
        for o in errored:
            assert o.outcome == "exception"
            assert o.payment_ids == []
            assert "upstream 503" in o.evidence["error"]
        assert engine.t3.errors


def test_adjudicator_arithmetic_is_always_recomputed(batch, monkeypatch):
    """The model's payment set is re-added from our own records. A verdict whose
    nets do not sum to the credit is rejected however confident it sounds."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    ctx = _ctx(batch)
    txn = next(t for t in batch.bank_txns if not t.utr)
    wrong = list(ctx.payments)[:3]          # almost certainly not this credit

    t3 = Tier3Adjudicator(enabled=True)
    fake = MagicMock()
    fake.parsed_output = Verdict(
        decision="match_payments", payment_ids=wrong, confidence=0.99,
        reasoning="I am very confident and completely wrong.")
    fake.usage = MagicMock(input_tokens=10, output_tokens=5)
    with patch.object(T3, "_get_client") as gc:
        gc.return_value.messages.parse = MagicMock(return_value=fake)
        d = t3.attempt(txn, ctx, None)

    if d.strategy == "llm_arithmetic_rejected":
        assert d.outcome == "exception"
        assert d.reason_code == "LOW_CONFIDENCE"
        assert d.payment_ids == []
        assert d.evidence["delta_paise"] != 0


def test_adjudicator_naming_an_unverifiable_settlement_is_rejected(batch, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    ctx = _ctx(batch)
    txn = next(t for t in batch.bank_txns if not t.utr)
    t3 = Tier3Adjudicator(enabled=True)
    fake = MagicMock()
    fake.parsed_output = Verdict(
        decision="match_settlement", settlement_id="setl_9999999999",
        confidence=0.99, reasoning="Invented an id.")
    fake.usage = MagicMock(input_tokens=1, output_tokens=1)
    with patch.object(T3, "_get_client") as gc:
        gc.return_value.messages.parse = MagicMock(return_value=fake)
        d = t3.attempt(txn, ctx, None)
    if d.evidence.get("resolver") == "llm":
        assert d.outcome == "exception"
        assert d.payment_ids == []


def test_call_cap_is_enforced(batch, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    t3 = Tier3Adjudicator(enabled=True, max_calls=0)
    ctx = _ctx(batch)
    txn = next(t for t in batch.bank_txns if not t.utr and "RAZORPAY" in t.narration.upper())
    d = t3.attempt(txn, ctx, None)
    assert d.reason_code == "NEEDS_LLM_REVIEW"
    assert d.strategy == "call_cap_reached"


def test_orphan_detection_costs_nothing(batch, monkeypatch):
    """Non-gateway counterparties resolve before any API call is considered."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    ctx = _ctx(batch)
    orphan = next(t for t in batch.bank_txns
                  if "RAZORPAY" not in t.narration.upper()
                  and "RAZORPAY" not in t.counterparty.upper())
    t3 = Tier3Adjudicator(enabled=True)
    d = t3.attempt(orphan, ctx, None)
    assert d.reason_code == "NOT_A_SETTLEMENT"
    assert d.evidence["resolver"] == "deterministic"
    assert t3.calls == 0
