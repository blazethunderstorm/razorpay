"""Exact subset reconstruction for bundled settlements.

The core question: a bank credit of Rs 3,47,182.16 arrived. Which of the 23
open payments, net of MDR and GST, add up to exactly that? This is subset-sum,
and a hackathon-grade answer would call an LLM and hope. We solve it exactly.

Five strategies, tried cheapest-first, with automatic selection:

  1. ``full_pool``      -- O(n).      The whole open pool settled together.
  2. ``complement_k``   -- O(n^k).    Everything except k stragglers. This is the
                                      shape most real cycles actually have.
  3. ``forward_k``      -- O(n^k).    Only k payments settled (instant payouts).
  4. ``meet_in_middle`` -- O(2^(n/2)) Exact and complete for pools up to 38.
  5. ``bitset_dp``      -- feasibility via Python bigint shifts, then a capped
                           dict DP to reconstruct. The safety net for pools
                           too large to halve.

Two properties matter more than speed:

**Completeness.** Strategies 1-4 enumerate their entire space, so when they
return one solution we know it is the only one of that shape. That is what
lets us distinguish "confidently unique" from "ambiguous".

**Honest ambiguity.** Two solutions that use different payment *ids* but the
same multiset of *amounts* are economically identical -- the cash attribution
is the same and only the id labelling differs, so we canonicalise and proceed.
Two solutions with genuinely different amount multisets mean the credit has
more than one valid explanation, and the engine must refuse to post rather
than guess. Guessing here silently misstates which orders were paid for.
"""

from __future__ import annotations

import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from itertools import combinations


@dataclass(frozen=True)
class Item:
    key: str
    value: int   # paise; negative for refunds/chargebacks netted off the payout


@dataclass(frozen=True)
class Solution:
    keys: tuple[str, ...]
    total: int
    residual: int

    @property
    def signature(self) -> tuple[int, ...]:
        return tuple(sorted(self._values))

    _values: tuple[int, ...] = field(default=(), compare=False)


@dataclass
class SolverResult:
    solutions: list[Solution]
    strategy: str
    explored: int
    elapsed_ms: float
    complete: bool          # did the winning strategy enumerate its full space?
    budget_exceeded: bool
    pool_size: int
    note: str = ""

    @property
    def distinct_signatures(self) -> int:
        return len({s.signature for s in self.solutions})

    @property
    def ambiguous(self) -> bool:
        return self.distinct_signatures > 1

    @property
    def best(self) -> Solution | None:
        if not self.solutions:
            return None
        # Deterministic tie-break among economically identical solutions:
        # fewest legs, then lexicographically smallest id set. Stable across
        # runs, which matters because the ledger must be reproducible.
        return min(self.solutions, key=lambda s: (abs(s.residual), len(s.keys), s.keys))


MITM_EXACT_N = 30       # always worth doing exactly -- 2^15 per half is milliseconds
MITM_MAX_N = 40         # still attemptable, but only after the cheap paths miss
BITSET_TARGET_CAP = 200_000_000     # Rs 20 lakh in paise
DP_STATE_CAP = 1_500_000
MAX_COLLECTED = 64      # stop hoarding id-permutations of the same amount shape


def _mk(keys: tuple[str, ...], values: tuple[int, ...], target: int) -> Solution:
    total = sum(values)
    return Solution(keys=tuple(sorted(keys)), total=total,
                    residual=total - target, _values=values)


def solve(items: list[Item], target: int, tolerance: int = 0,
          max_solutions: int = 4, budget_ms: int = 2000) -> SolverResult:
    """Find payment subsets whose net total reaches ``target`` +/- ``tolerance``.

    ``SolverResult.complete`` is the field that matters downstream. True means
    the strategy enumerated its entire space, so a single returned solution is
    provably the only one and the engine may post it. False means we found
    *a* solution but cannot rule out others, and the engine must treat it as
    unverified rather than certified.
    """
    t0 = time.perf_counter()
    explored = 0

    def elapsed() -> float:
        return (time.perf_counter() - t0) * 1000.0

    def out(sols: list[Solution], strategy: str, complete: bool,
            note: str = "", over: bool = False) -> SolverResult:
        seen: set[tuple[str, ...]] = set()
        uniq: list[Solution] = []
        for s in sols:
            if s.keys not in seen:
                seen.add(s.keys)
                uniq.append(s)
        # Keep at least one representative of every distinct amount shape, so
        # truncation can never hide an ambiguity from the caller.
        by_sig: dict[tuple[int, ...], Solution] = {}
        for s in uniq:
            by_sig.setdefault(s.signature, s)
        kept = list(by_sig.values())
        for s in uniq:
            if len(kept) >= max_solutions:
                break
            if s not in kept:
                kept.append(s)
        return SolverResult(
            solutions=kept[:max(max_solutions, len(by_sig))], strategy=strategy,
            explored=explored, elapsed_ms=round(elapsed(), 2), complete=complete,
            budget_exceeded=over, pool_size=len(items), note=note,
        )

    if not items:
        return out([], "empty_pool", True, "No open payments in the search window.")

    has_negative = any(i.value < 0 for i in items)

    # Prune items that cannot appear in any solution. Only sound when every
    # value is non-negative -- with refunds in the pool a large payment can be
    # offset back under target, so the prune would discard real answers.
    pool = list(items)
    pruned = 0
    if not has_negative:
        pool = [i for i in pool if i.value <= target + tolerance]
        pruned = len(items) - len(pool)
        if not pool:
            return out([], "prefilter", True,
                       "Every open payment individually exceeds the credit amount.")

    total_all = sum(i.value for i in pool)
    # Feasibility bounds. With refunds in the pool the reachable maximum is the
    # sum of the *positive* values, not the sum of everything -- bounding by the
    # net would declare perfectly solvable searches impossible the moment a
    # large refund is in the window.
    reach_max = sum(i.value for i in pool if i.value > 0)
    reach_min = sum(i.value for i in pool if i.value < 0)
    if reach_max < target - tolerance:
        return out([], "prefilter", True,
                   f"Open pool tops out at {reach_max} paise, below the {target} paise credit.")
    if target < reach_min - tolerance:
        return out([], "prefilter", True,
                   f"Credit of {target} paise is below the pool floor of {reach_min} paise.")

    n = len(pool)
    pnote = f"{pruned} payment(s) pruned as individually larger than the credit. " if pruned else ""

    # --- exact and complete, whenever the pool is small enough ----------
    if n <= MITM_EXACT_N:
        sols, ex, complete = _meet_in_middle(pool, target, tolerance, t0, budget_ms)
        explored += ex
        if sols:
            return out(sols, "meet_in_middle", complete, pnote.strip())
        return out([], "meet_in_middle", complete,
                   pnote + "Exhaustive search: no subset of the open pool reaches this amount.")

    # --- large pools: cheap shapes first --------------------------------
    found: list[Solution] = []
    explored += 1
    if abs(total_all - target) <= tolerance:
        found.append(_mk(tuple(i.key for i in pool), tuple(i.value for i in pool), target))
        return out(found, "full_pool", False,
                   pnote + "Whole open pool settled together; uniqueness not proven at this pool size.")

    need = total_all - target
    for k in (1, 2, 3):
        if k >= n or elapsed() > budget_ms:
            break
        for combo in combinations(range(n), k):
            explored += 1
            if abs(sum(pool[i].value for i in combo) - need) <= tolerance:
                keep = [pool[i] for i in range(n) if i not in combo]
                found.append(_mk(tuple(i.key for i in keep),
                                 tuple(i.value for i in keep), target))
                if len(found) >= MAX_COLLECTED:
                    break
        if found:
            return out(found, f"complement_{k}", False,
                       pnote + f"Pool minus {k}; uniqueness not proven at this pool size.")

    for k in (1, 2, 3):
        if k > n or elapsed() > budget_ms:
            break
        for combo in combinations(range(n), k):
            explored += 1
            vals = tuple(pool[i].value for i in combo)
            if abs(sum(vals) - target) <= tolerance:
                found.append(_mk(tuple(pool[i].key for i in combo), vals, target))
                if len(found) >= MAX_COLLECTED:
                    break
        if found:
            return out(found, f"forward_{k}", False,
                       pnote + f"{k} payment(s) settled alone; uniqueness not proven.")

    # --- still nothing: pay for the exact search if we can afford it ----
    if n <= MITM_MAX_N and elapsed() < budget_ms * 0.5:
        sols, ex, complete = _meet_in_middle(pool, target, tolerance, t0, budget_ms)
        explored += ex
        if sols:
            return out(sols, "meet_in_middle", complete, pnote.strip())
        if complete:
            return out([], "meet_in_middle", True,
                       pnote + "Exhaustive search: no subset of the open pool reaches this amount.")

    # --- last resort: bitset feasibility, then capped reconstruction ----
    if not has_negative and target <= BITSET_TARGET_CAP:
        if not _bitset_feasible(pool, target, tolerance):
            return out([], "bitset_dp", True,
                       pnote + "Bitset reachability proof: no subset can reach this amount.")
        sols, states = _dict_dp(pool, target, tolerance, max_solutions)
        explored += states
        over = states >= DP_STATE_CAP
        return out(sols, "bitset_dp", False,
                   pnote + (f"Reachable-state cap of {DP_STATE_CAP:,} hit." if over
                            else "Reconstructed by DP; uniqueness not proven."), over)

    return out([], "budget_exceeded", False,
               pnote + f"Pool of {n} exceeds exact-search limits for this shape.", True)


def _meet_in_middle(pool: list[Item], target: int, tolerance: int,
                    t0: float, budget_ms: int) -> tuple[list[Solution], int, bool]:
    """Split the pool, enumerate both halves, join on the sorted left sums.

    Enumerates the complete 2^n space in 2^(n/2) time and space, so a result
    from here is authoritative: every solution that exists is found, which is
    exactly what ambiguity detection requires.
    """
    n = len(pool)
    half = n // 2
    left, right = pool[:half], pool[half:]

    def enumerate_half(items: list[Item]) -> list[tuple[int, int]]:
        acc: list[tuple[int, int]] = [(0, 0)]
        for idx, it in enumerate(items):
            bit = 1 << idx
            acc += [(s + it.value, m | bit) for s, m in acc]
        return acc

    lo = enumerate_half(left)
    hi = enumerate_half(right)
    explored = len(lo) + len(hi)

    lo.sort(key=lambda x: x[0])
    lo_sums = [s for s, _ in lo]

    out: list[Solution] = []
    sigs: set[tuple[int, ...]] = set()
    for rs, rmask in hi:
        want = target - rs
        i = bisect_left(lo_sums, want - tolerance)
        j = bisect_right(lo_sums, want + tolerance)
        for idx in range(i, j):
            ls, lmask = lo[idx]
            if lmask == 0 and rmask == 0:
                continue     # the empty set is not an explanation
            keys, vals = [], []
            for b in range(len(left)):
                if lmask >> b & 1:
                    keys.append(left[b].key)
                    vals.append(left[b].value)
            for b in range(len(right)):
                if rmask >> b & 1:
                    keys.append(right[b].key)
                    vals.append(right[b].value)
            sol = _mk(tuple(keys), tuple(vals), target)
            out.append(sol)
            sigs.add(sol.signature)
            if len(out) >= MAX_COLLECTED:
                # Ambiguity is already established or the pool is degenerate;
                # either way the caller will not auto-post, so stop early.
                return out, explored, False
        if (time.perf_counter() - t0) * 1000.0 > budget_ms:
            return out, explored, False
    return out, explored, True

def _bitset_feasible(pool: list[Item], target: int, tolerance: int) -> bool:
    """Reachability via one big integer: bit i set == sum i is achievable.

    ``mask |= mask << v`` shifts every reachable sum up by v in a single
    C-level bigint operation, so the whole DP costs n shift-ors instead of
    n*target Python-level steps.
    """
    cap = target + tolerance
    limit = (1 << (cap + 1)) - 1
    mask = 1
    for it in pool:
        mask |= mask << it.value
        mask &= limit
    window = mask >> max(0, target - tolerance)
    return bool(window & ((1 << (2 * tolerance + 1)) - 1))


def _dict_dp(pool: list[Item], target: int, tolerance: int,
             max_solutions: int) -> tuple[list[Solution], int]:
    """Reachable-sum DP with parent pointers, capped so it cannot run away."""
    parent: dict[int, tuple[int, int]] = {0: (-1, -1)}
    for idx, it in enumerate(pool):
        for s in list(parent.keys()):
            ns = s + it.value
            if ns > target + tolerance or ns in parent:
                continue
            parent[ns] = (s, idx)
            if len(parent) >= DP_STATE_CAP:
                return _reconstruct(parent, pool, target, tolerance, max_solutions), len(parent)
    return _reconstruct(parent, pool, target, tolerance, max_solutions), len(parent)


def _reconstruct(parent: dict[int, tuple[int, int]], pool: list[Item],
                 target: int, tolerance: int, max_solutions: int) -> list[Solution]:
    out: list[Solution] = []
    for cand in range(target - tolerance, target + tolerance + 1):
        if cand not in parent or cand == 0:
            continue
        keys, vals, cur = [], [], cand
        while cur != 0:
            prev, idx = parent[cur]
            if idx < 0:
                break
            keys.append(pool[idx].key)
            vals.append(pool[idx].value)
            cur = prev
        if keys:
            out.append(_mk(tuple(keys), tuple(vals), target))
        if len(out) >= max_solutions:
            break
    return out
