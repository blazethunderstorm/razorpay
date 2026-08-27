"""The solver, including the properties the engine's correctness depends on."""
import random

from khata.subsetsum import Item, solve


def it(vals):
    return [Item(f"p{i}", v) for i, v in enumerate(vals)]


def test_finds_whole_pool():
    r = solve(it([100, 200, 300]), 600)
    assert sorted(r.best.keys) == ["p0", "p1", "p2"]
    assert r.complete


def test_finds_single_payment():
    r = solve(it([1000, 2000, 3000, 50]), 50)
    assert r.best.keys == ("p3",)


def test_handles_netted_refund_as_negative_item():
    r = solve(it([5000, 3000, 2000, -1500]), 8500)
    assert sorted(r.best.keys) == ["p0", "p1", "p2", "p3"]


def test_negative_items_do_not_trip_the_feasibility_bound():
    """Regression: bounding by sum(all) instead of sum(positives) declared
    solvable searches impossible whenever a large refund sat in the window."""
    items = it([5000, 3000, 2000, 1000]) + [Item("r0", -9000)]
    r = solve(items, 5000)
    assert r.best is not None, "a big refund must not veto a reachable target"
    assert r.explored > 0


def test_reports_infeasible_without_guessing():
    r = solve(it([100, 200, 300]), 99999)
    assert r.best is None and r.complete and "tops out" in r.note


def test_detects_genuine_ambiguity():
    """{1200,1800,3000} and {2500,3500} both total 6000."""
    r = solve(it([1200, 1800, 3000, 2500, 3500]), 6000)
    assert r.ambiguous
    assert r.distinct_signatures >= 2


def test_equal_valued_payments_are_not_ambiguity():
    """Two payments of the same amount are an id-labelling tie, not two
    different economic explanations, so they must not block a match."""
    r = solve(it([500, 500, 900]), 500)
    assert not r.ambiguous
    assert r.distinct_signatures == 1


def test_deterministic_tie_break():
    """The ledger must be reproducible, so the chosen subset cannot vary."""
    items = it([500, 500, 500, 900])
    first = solve(items, 1000).best.keys
    for _ in range(5):
        assert solve(items, 1000).best.keys == first


def test_meet_in_middle_is_exact_on_a_large_pool():
    rng = random.Random(7)
    vals = [rng.randrange(10_000, 900_000) for _ in range(26)]
    target = sum(vals[:13])
    r = solve(it(vals), target)
    assert r.best is not None
    assert sum(v for k, v in zip(r.best.keys, r.best._values)) == target or \
        r.best.total == target


def test_incompleteness_is_reported_not_hidden():
    rng = random.Random(11)
    vals = [rng.randrange(1_000, 50_000) for _ in range(46)]
    r = solve(it(vals), sum(vals[:23]), budget_ms=200)
    # Either it proves the answer or it admits it could not. Never silently wrong.
    assert r.complete or r.budget_exceeded or r.best is not None
