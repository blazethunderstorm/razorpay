"""The exact-cover partition pass.

Kept, tested, and off by default. It is correct but on realistic pool sizes it
has never proved a unique partition -- the tests below pin both the correctness
and that honest limitation.
"""
from khata.engine import Engine
from khata.evaluate import evaluate
from khata.generator import generate
from khata.groupsolve import CANDIDATES_PER_TARGET, solve_partition
from khata.subsetsum import Item


def it(vals):
    return [Item(f"p{i}", v) for i, v in enumerate(vals)]


def test_needs_at_least_two_credits():
    r = solve_partition(it([100, 200]), [300])
    assert not r.unique and "two credits" in r.note


def test_infeasible_day_is_proven_not_guessed():
    r = solve_partition(it([100, 200]), [5000, 5000])
    assert r.assignments == [] and r.complete
    assert "short of" in r.note


def test_unique_partition_is_found():
    r = solve_partition(it([1000, 2000, 3000, 1500, 4500]), [6000, 6000])
    assert r.unique
    picks = {i: sorted(sol.keys) for i, sol in r.assignments[0].picks}
    assert sorted(picks[0] + picks[1]) == ["p0", "p1", "p2", "p3", "p4"]


def test_swapping_equal_amount_credits_is_not_a_second_answer():
    """Two credits for the same amount on the same day are interchangeable."""
    r = solve_partition(it([100, 100, 100, 100]), [100, 100])
    assert r.unique


def test_genuinely_different_partitions_stay_ambiguous():
    # {3000} + {1000,2000,4000}  and  {1000,2000} + {3000,4000}
    r = solve_partition(it([1000, 2000, 3000, 4000]), [3000, 7000])
    assert r.ambiguous and not r.unique


def test_uniqueness_is_never_claimed_from_a_truncated_search():
    """Regression. Enumerating only the first N subsets per credit and then
    calling the single surviving assignment 'unique' posted two false matches
    before this guard existed: the assignment that would have proved ambiguity
    was simply never enumerated."""
    # Many equal-valued items give each target far more than N candidate subsets.
    items = it([100] * 24)
    r = solve_partition(items, [400, 400])
    if len(r.assignments) == 1:
        assert not r.complete, "a capped enumeration cannot prove uniqueness"
        assert not r.unique
        assert "truncated" in r.note.lower()


def test_group_pass_is_off_by_default():
    batch = generate(seed=42, n_payments=300, days=12)
    result = Engine(use_llm=False).run(batch)
    assert result.group["enabled"] is False
    assert result.group["credits_upgraded"] == 0


def test_group_pass_adds_no_false_match_of_its_own():
    """Enabling the pass must not make precision worse than leaving it off.

    Stated as a comparison rather than as `false_matches == 0` on purpose: an
    absolute assertion here would fail for reasons that have nothing to do with
    this pass and misattribute the blame, which is exactly what it did while
    being written.
    """
    for seed in (42, 900):
        batch = generate(seed=seed, n_payments=900, days=21)
        on_run = Engine(use_llm=False, gateway_budget=0, enable_group=True).run(batch)
        on = evaluate(batch, on_run)
        off = evaluate(batch, Engine(use_llm=False, gateway_budget=0,
                                     enable_group=False).run(batch))
        assert on.false_matches <= off.false_matches, f"seed {seed}"
        assert on.line_recall >= off.line_recall
        for o in on_run.outcomes:
            if o.tier == "T4-GROUP":
                assert o.evidence["search_complete"], "posted an unproven partition"
