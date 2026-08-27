"""Money arithmetic and the fee model.

These are the foundation: if MDR rounding is off by a paise, subset-sum
reconstruction stops working and every downstream metric is meaningless.
"""
from decimal import Decimal

from khata.fees import GST_RATE, compute_fee, net_of
from khata.money import fmt, pct, rupees


def test_rupees_rounds_half_up():
    assert rupees("10.005") == 1001
    assert rupees(1) == 100
    assert rupees("0.01") == 1


def test_indian_number_grouping():
    assert fmt(4732811) == "₹47,328.11"
    assert fmt(100000000) == "₹10,00,000.00"
    assert fmt(-1234567890) == "-₹1,23,45,678.90"
    assert fmt(50) == "₹0.50"


def test_upi_is_zero_mdr():
    fb = compute_fee(rupees(1000), "upi")
    assert (fb.mdr_paise, fb.gst_paise) == (0, 0)
    assert fb.net_paise == fb.gross_paise


def test_card_fee_and_gst_on_fee_not_on_value():
    fb = compute_fee(rupees(1000), "card")
    assert fb.mdr_paise == rupees(20)                    # 2.00%
    assert fb.gst_paise == int(Decimal(fb.mdr_paise) * GST_RATE)
    assert fb.net_paise == fb.gross_paise - fb.mdr_paise - fb.gst_paise


def test_fee_components_never_lose_a_paise():
    for amount in (1, 99, 12345, 999999, 4999999):
        for method in ("upi", "card", "netbanking", "wallet"):
            fb = compute_fee(amount, method)
            assert fb.net_paise + fb.mdr_paise + fb.gst_paise == fb.gross_paise
            assert net_of(amount, method) == fb.net_paise


def test_pct_survives_empty_batch():
    assert pct(0, 0) == 0.0
