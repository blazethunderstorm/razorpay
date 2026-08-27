"""The ledger invariant. If this ever fails, the run is void."""
import pytest

from khata.ledger import Ledger, LedgerImbalance


def test_rejects_unbalanced_posting():
    l = Ledger()
    with pytest.raises(LedgerImbalance):
        l.post("bad", "x", {"bank": 100}, {"revenue": 99})
    assert l.postings == []          # nothing partially applied


def test_rejects_unknown_account():
    l = Ledger()
    with pytest.raises(KeyError):
        l.post("bad", "x", {"not_an_account": 100}, {"revenue": 100})


def test_capture_then_settle_clears_the_clearing_account():
    l = Ledger()
    l.payment_captured("pay_1", 100000, 2000, 360, 97640)
    assert l.balances["gateway_clearing"] == 97640
    l.settlement_matched("bnk_1", 97640)
    assert l.balances["gateway_clearing"] == 0
    assert l.balances["bank"] == 97640
    l.assert_balanced()


def test_unattributed_cash_lands_in_suspense_not_clearing():
    l = Ledger()
    l.credit_unattributed("bnk_9", 5000)
    assert l.suspense_balance() == 5000
    assert l.balances["gateway_clearing"] == 0
    l.assert_balanced()


def test_refund_and_chargeback_reduce_clearing():
    l = Ledger()
    l.payment_captured("pay_1", 100000, 0, 0, 100000)
    l.refund_issued("rfnd_1", 40000)
    l.chargeback_raised("disp_1", 10000)
    assert l.balances["gateway_clearing"] == 50000
    l.assert_balanced()
