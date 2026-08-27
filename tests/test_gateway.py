"""The budgeted gateway client.

The client must be expensive, countable, and non-oracular. If it ever becomes a
way to look up the answer without first identifying the settlement, every metric
downstream stops meaning anything.
"""
from datetime import datetime

import pytest

from khata.engine import Engine
from khata.evaluate import evaluate
from khata.gateway import GatewayClient, NullGateway
from khata.generator import generate
from khata.models import SettlementAdvice


def _advice(sid="setl_1", net=1000, pids=("pay_1", "pay_2")):
    return SettlementAdvice(sid, "UTR1", net, datetime(2026, 7, 3), list(pids),
                            breakup_available=False, record_available=False)


def test_fetch_returns_the_breakup():
    g = GatewayClient(ledger=[_advice()], budget=5)
    assert g.fetch_recon("setl_1", "bnk_1") == ["pay_1", "pay_2"]
    assert g.spent == 1


def test_budget_is_hard():
    g = GatewayClient(ledger=[_advice()], budget=2)
    assert g.fetch_recon("setl_1", "bnk_1")
    assert g.fetch_recon("setl_1", "bnk_2")
    assert g.fetch_recon("setl_1", "bnk_3") is None, "budget must refuse, not stretch"
    assert g.remaining == 0
    assert g.summary()["refused"] == 1


def test_every_call_records_who_justified_it():
    g = GatewayClient(ledger=[_advice()], budget=5)
    g.fetch_recon("setl_1", "bnk_77")
    assert g.calls[0].on_behalf_of == "bnk_77"
    assert g.calls[0].op == "fetch_recon"


def test_unknown_settlement_still_costs_a_call():
    g = GatewayClient(ledger=[_advice()], budget=5)
    assert g.fetch_recon("setl_nope", "bnk_1") is None
    assert g.spent == 1, "a miss is still a request"


def test_list_returns_no_payment_detail():
    a = _advice()
    g = GatewayClient(ledger=[a], budget=5)
    rows = g.list_settlements(a.settled_at.date(), "bnk_1")
    assert rows and "payment_ids" not in rows[0]
    assert set(rows[0]) == {"settlement_id", "net_paise", "utr", "settled_at"}


def test_null_gateway_never_reaches_out():
    g = NullGateway()
    assert g.fetch_recon("setl_1", "bnk_1") is None
    assert g.list_settlements(datetime(2026, 7, 3).date(), "bnk_1") == []


def test_fetch_is_not_an_oracle():
    """Pointed at the wrong settlement it returns the wrong payments, happily.
    Attribution precision is what makes the fetch safe, not the fetch itself."""
    a1 = _advice("setl_1", 1000, ("pay_1",))
    a2 = _advice("setl_2", 1000, ("pay_2",))
    g = GatewayClient(ledger=[a1, a2], budget=5)
    assert g.fetch_recon("setl_2", "bnk_1") == ["pay_2"]


@pytest.mark.parametrize("budget", [0, 40])
def test_engine_respects_the_budget_and_stays_precise(budget):
    batch = generate(seed=42, n_payments=300, days=12)
    result = Engine(use_llm=False, gateway_budget=budget).run(batch)
    m = evaluate(batch, result)
    assert result.gateway["spent"] <= budget
    assert result.ledger.trial_balance()["balanced"]
    assert m.false_matches == 0


def test_gateway_recovers_line_detail_that_arithmetic_cannot():
    """The headline claim for this capability: it converts cash-only into lines."""
    batch = generate(seed=42)
    off = evaluate(batch, Engine(use_llm=False, gateway_budget=0).run(batch))
    on = evaluate(batch, Engine(use_llm=False, gateway_budget=40).run(batch))
    assert on.line_recall > off.line_recall
    assert on.cash_only < off.cash_only
    assert on.false_matches == 0


def test_a_batch_is_not_mutated_by_running_it():
    """Regression: recovering a breakup wrote it back onto the batch's own
    advice objects, so a second run over the same batch found the work already
    done and reported roughly half the API cost. Repeat runs must cost the same."""
    batch = generate(seed=42)
    before = sum(a.breakup_available for a in batch.advices)
    first = Engine(use_llm=False, gateway_budget=40).run(batch)
    after = sum(a.breakup_available for a in batch.advices)
    assert after == before, "the engine mutated the batch it was given"

    second = Engine(use_llm=False, gateway_budget=40).run(batch)
    assert second.gateway["spent"] == first.gateway["spent"]
    assert evaluate(batch, second).line_recall == evaluate(batch, first).line_recall
