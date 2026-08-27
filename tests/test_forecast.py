"""Forward cash forecast.

The forecast makes exactly one promise -- that it never guesses -- so the tests
are mostly about what it refuses to include.
"""
from datetime import date, timedelta

from khata.forecast import SETTLEMENT_LAG_DAYS, forecast
from khata.generator import generate


def batch():
    return generate(seed=42, n_payments=600, days=21)


def test_defaults_to_last_capture_date():
    b = batch()
    f = forecast(b)
    assert f.as_of == max(p.captured_at.date() for p in b.payments)


def test_only_includes_cycles_after_as_of():
    b = batch()
    f = forecast(b)
    assert f.days, "seed 42 should have captures still awaiting payout"
    assert all(d.payout_date > f.as_of for d in f.days)


def test_payout_is_capture_plus_lag():
    b = batch()
    f = forecast(b)
    for d in f.days:
        assert d.payout_date == d.capture_date + timedelta(days=f.lag_days)


def test_horizon_is_respected():
    b = batch()
    f = forecast(b, horizon=1)
    assert all(d.payout_date <= f.as_of + timedelta(days=1) for d in f.days)


def test_expected_is_gross_net_of_everything():
    b = batch()
    for d in forecast(b).days:
        assert d.expected_paise == (d.gross_paise - d.mdr_paise - d.gst_paise
                                    - d.refunds_paise - d.chargebacks_paise)


def test_a_refund_only_cycle_is_a_debit():
    """A cycle with no captures but an open chargeback nets negative, and the
    forecast must show that rather than clamping it to zero."""
    b = batch()
    days = forecast(b).days
    debits = [d for d in days if d.expected_paise < 0]
    assert debits, "seed 42 has chargeback-only cycles"
    assert all(d.payments == 0 for d in debits)


def test_lag_is_configurable():
    b = batch()
    f = forecast(b, lag_days=3)
    assert f.lag_days == 3
    assert all(d.payout_date == d.capture_date + timedelta(days=3) for d in f.days)


def test_empty_when_nothing_outstanding():
    """As of a date past every payout, there is nothing left to forecast."""
    b = batch()
    late = max(p.captured_at.date() for p in b.payments) + timedelta(days=30)
    f = forecast(b, as_of=late)
    assert f.days == [] and f.total_paise == 0


def test_default_lag_matches_the_generator():
    assert SETTLEMENT_LAG_DAYS == 2
