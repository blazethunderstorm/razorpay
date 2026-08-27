"""Exact-cover partition: reconcile several credits from one capture day together.

Tier 2 asks a narrow question -- "which payments make up *this* credit" -- and
answers it one credit at a time. That is why a busy capture day defeats it: with
thirty payments open, several different subsets total the same amount, Tier 2
proves the ambiguity and correctly refuses.

But the day's credits are not independent. Three settlement cycles landing on the
same date partition the *same* pool between them, and no payment can appear
twice. Asking the wider question -- "is there an assignment of disjoint subsets
that explains *all* of today's credits at once" -- adds a constraint strong
enough to collapse ambiguity that is genuinely unresolvable credit-by-credit.
A decomposition that balances alone but leaves a sibling unexplainable was never
the right one.

The uniqueness rule from Tier 2 carries over unchanged, one level up: a single
valid assignment is posted, two or more distinct assignments means the day does
not determine the answer and nothing is posted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .subsetsum import Item, Solution, solve

CANDIDATES_PER_TARGET = 16
MAX_TARGETS = 5
NODE_BUDGET = 20_000


@dataclass
class Assignment:
    """One complete explanation of a day: target index -> chosen subset."""

    picks: list[tuple[int, Solution]] = field(default_factory=list)

    def signature(self, targets: list[int]) -> tuple:
        """Canonical form, keyed on amounts rather than on which credit got what.

        Two credits for the same amount on the same date are interchangeable --
        nothing in the data says which subset belongs to which statement line, so
        swapping them is a relabelling, not a second explanation. Keying the
        signature on (target amount, chosen value multiset) collapses those
        permutations while keeping genuinely different partitions distinct. This
        is the same rule Tier 2 applies to equal-valued payments, one level up.
        """
        return tuple(sorted((targets[i], sol.signature) for i, sol in self.picks))

    def keys_for(self, idx: int) -> tuple[str, ...]:
        return next(sol.keys for i, sol in self.picks if i == idx)


@dataclass
class PartitionResult:
    assignments: list[Assignment]
    explored: int
    elapsed_ms: float
    complete: bool
    note: str = ""

    @property
    def unique(self) -> bool:
        """One assignment, *and* the search that found it was exhaustive.

        Claiming uniqueness from a truncated candidate list is the trap this
        whole module walked into: enumerate only the first 16 subsets per credit
        and the assignment that would have proved ambiguity is simply never
        seen, so a coin flip gets posted as a proof. Uniqueness that cannot be
        demonstrated is not uniqueness.
        """
        return len(self.assignments) == 1 and self.complete

    @property
    def ambiguous(self) -> bool:
        return len(self.assignments) > 1


def solve_partition(items: list[Item], targets: list[int],
                    tolerance: int = 0, budget_ms: int = 1500,
                    max_assignments: int = 2) -> PartitionResult:
    """Find disjoint subsets of ``items`` summing to each of ``targets``.

    Stops as soon as ``max_assignments`` distinct assignments are found, because
    two is already enough to prove the day is ambiguous and nothing will be
    posted. Searching for a third would cost time to learn nothing.
    """
    t0 = time.perf_counter()
    k = len(targets)
    if k < 2:
        return PartitionResult([], 0, 0.0, True,
                               "Partitioning needs at least two credits.")
    if k > MAX_TARGETS:
        return PartitionResult([], 0, 0.0, False,
                               f"{k} credits on one day exceeds the {MAX_TARGETS}-target limit.")

    reachable = sum(i.value for i in items if i.value > 0)
    if sum(targets) - tolerance * k > reachable:
        return PartitionResult(
            [], 0, round((time.perf_counter() - t0) * 1000, 2), True,
            f"The day's open payments total {reachable} paise, short of the "
            f"{sum(targets)} paise these credits require together.")

    # Per-target candidate subsets, computed once against the full pool.
    cands: list[list[Solution]] = []
    explored = 0
    truncated: list[int] = []
    for tgt in targets:
        r = solve(items, tgt, tolerance=tolerance,
                  max_solutions=CANDIDATES_PER_TARGET,
                  budget_ms=max(200, budget_ms // (2 * k)))
        explored += r.explored
        if not r.solutions:
            return PartitionResult(
                [], explored, round((time.perf_counter() - t0) * 1000, 2), r.complete,
                f"No subset of the day reaches {tgt} paise, so no partition exists.")
        # A capped or budget-limited enumeration cannot support a uniqueness
        # claim later, so record it now rather than discovering it too late.
        if not r.complete or len(r.solutions) >= CANDIDATES_PER_TARGET:
            truncated.append(tgt)
        cands.append(r.solutions)

    # Most-constrained-first: the target with fewest options is decided earliest,
    # so dead branches die shallow instead of after k-1 wasted levels.
    order = sorted(range(k), key=lambda i: len(cands[i]))
    found: list[Assignment] = []
    seen: set[tuple] = set()
    nodes = 0

    def out_of_time() -> bool:
        return (time.perf_counter() - t0) * 1000 > budget_ms

    def recurse(depth: int, used: frozenset[str], picks: list) -> bool:
        """Returns True when the caller should stop searching entirely."""
        nonlocal nodes
        if depth == len(order):
            a = Assignment(picks=list(picks))
            sig = a.signature(targets)
            if sig not in seen:
                seen.add(sig)
                found.append(a)
            return len(found) >= max_assignments
        idx = order[depth]
        for sol in cands[idx]:
            nodes += 1
            if nodes > NODE_BUDGET or out_of_time():
                return True
            keys = set(sol.keys)
            if keys & used:
                continue
            picks.append((idx, sol))
            stop = recurse(depth + 1, used | keys, picks)
            picks.pop()
            if stop:
                return True
        return False

    stopped_early = recurse(0, frozenset(), [])
    complete = (not stopped_early and nodes <= NODE_BUDGET
                and not out_of_time() and not truncated)

    note = ""
    if truncated and len(found) == 1:
        note = (f"One assignment found, but {len(truncated)} of {k} credits had their "
                "candidate subsets truncated, so uniqueness is unproven. Not posting.")
    elif not found:
        note = ("No assignment of disjoint subsets explains all of the day's credits."
                if complete else
                "Search budget reached before an assignment was found.")
    elif len(found) > 1:
        note = (f"{len(found)} distinct assignments explain the day equally well; "
                "the partition constraint is not strong enough to pick one.")
    return PartitionResult(found, explored + nodes,
                           round((time.perf_counter() - t0) * 1000, 2), complete, note)
