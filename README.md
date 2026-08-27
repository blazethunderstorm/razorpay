# खाता · Khata

**Settlement reconciliation for an Indian merchant on Razorpay, scored against held-out ground truth.**

Razorpay pays a merchant one bank credit for many payments, net of MDR and 18% GST on
that MDR, minus refunds and chargebacks raised in the same cycle. The merchant's
accountant has to work out *which* payments. By hand. Every day.

Khata closes that loop over a 500+ payment batch and reports what it could not solve.

> **Track 04 — AI Finance Controller.** The brief asks for one finance-ops loop closed
> across a 50+ record batch, with a match rate and an honest exception list.

---

## Results

Six independent seeds, ~300 bank credits, ~3,100 payments. Seeds 143–547 were never
looked at while building the matchers — only seed 42 was.

| | Cash precision | Cash recall | Line recall | False matches | Wrongly attributed | Wall | Cost |
|---|---|---|---|---|---|---|---|
| **Full pipeline** (4 seeds) | **100.0%** | **100.0%** | **98.3%** | **0** | **₹0.00** | 26 s | 15 LLM + 80 API |
| **No adjudicator** (6 seeds) | **100.0%** | 95.6% | 94.4% | **0** | **₹0.00** | 0.69 s | 0 LLM + ~20 API |
| **Fully offline** (6 seeds) | **100.0%** | 87.4% | 53.5% | **0** | **₹0.00** | 0.82 s | nothing |

```
   seed  credits   cash P   cash R   line R   FP       wrong Rs       ms   LLM
     42       47  100.0%  100.0%   95.2%    0          ₹0.00    31329     4
    143       50  100.0%  100.0%  100.0%    0          ₹0.00    23696     4
    244       53  100.0%  100.0%  100.0%    0          ₹0.00    12765     2
    345       53  100.0%  100.0%   97.9%    0          ₹0.00    36691     5
   mean       50  100.0%  100.0%   98.3%    0                   26120    15
```

**Zero false matches across 301 credits, at every batch size and volume tested.** That
is the number the whole design is bent around, and it is the only assertion allowed to
fail the build.

### What the two levels mean

A single "match rate" would hide which half failed, so results are scored twice:

- **Cash level** — is this bank credit attributed to the right settlement? This closes
  the books.
- **Line level** — do we know exactly which payments it comprises? This ages receivables
  by order. Strictly harder.

Precision and recall are kept apart on purpose. An engine hits 100% recall by matching
everything and 100% precision by matching nothing.

---

## Two negative results, kept because they are the interesting part

I finished a first version, wrote down what I thought the two right next steps were, then
built both and measured them. One returned 38 points. The other returned nothing and
briefly made things worse. Neither outcome was the one I expected.

### 1. The boring API call beat the clever algorithm

I described recovering the missing payment breakups as *"an API call, not an algorithm"* —
faintly dismissively. It is the single largest win in the project:

```
  configuration                       cash R  line R      d   eff R  cash P  FP  API  LLM      ms
  T0 only  (UTR lookup)               23.8%  23.8%   --    23.8% 100.0%   0    0    0       2
  +T1      (advice amount)            50.0%  50.0% +26.2   50.0% 100.0%   0    0    0       3
  +T2      (subset-sum)               54.8%  50.0%  +0.0   54.8% 100.0%   0    0    0     631
  +T2b     (cash attribution)         83.3%  50.0%  +0.0   54.8% 100.0%   0    0    0     618
  +gateway (recon fetch)              92.9%  88.1% +38.1   92.9% 100.0%   0   20    0     547
  +T4      (group partition)          92.9%  88.1%  +0.0   92.9% 100.0%   0    9    0     163
  +T3      (adjudicator)             100.0%  95.2%  +7.1  100.0% 100.0%   0    9    4   28554
```

Twenty budgeted API calls bought **+38.1 points of line recall**. The exact subset-sum
solver — by far the most interesting code here — bought **+0.0**.

`./run.sh ablate` reproduces it. Rows are cumulative, so deltas are order-dependent: each
tier is measured *given* the ones above it.

### 2. Exact-cover group partitioning: correct, and worth nothing

I said the sibling lookahead was a one-step approximation of an exact-cover problem over
all credits sharing a value date, and that solving them jointly was the real shape. It is
the real shape. It is also useless here, and it cost me two false matches on the way:

- **First version posted 2 false matches** (₹76,607 wrongly attributed). The bug was mine
  and it is a good one: I enumerated only the first 16 candidate subsets per credit, found
  one surviving assignment, and called it unique. The assignment that would have proved
  ambiguity was simply never enumerated. **Uniqueness claimed from a truncated search is
  not uniqueness.**
- **With the guard added, it now posts nothing at all** — at 139, 355, 778 and 1,216
  payments. It correctly *proves* ambiguity and never proves uniqueness, because a
  2–3 target partition over a 30-payment pool is not a strong enough constraint.

So it ships **off by default**, with its tests, and `run.sh ... --group` turns it on. A
slow pass that buys nothing does not belong in the default path, and deleting it would
hide the finding.

### 3. And the same bug class, twice more

Once I understood the shape of the truncation bug — *a claim of uniqueness resting on a
search that was never exhaustive* — it turned up in two more places:

- **Tier 2 skipped oversized capture days and still claimed cross-day uniqueness.** Above
  ~40 payments a day the true capture day exceeds the exact-search cap and is skipped; the
  single hit among the days we *did* search then looks decisive. It produced a 10-payment
  match where the truth was 15. Now a skipped day forces the score below the posting floor.
- **A ₹1,62,000 customer NEFT was posted as a 12-payment settlement** at 0.88 confidence.
  The "is this even a gateway credit?" check existed and was correct — it just lived in
  Tier 3, and Tier 2 got there first. Ordering was the entire defect. It is now `T0-SOURCE`
  and runs before anything else, because it is a precondition, not a tier.

Both were caught by parameterising the tests over batch sizes rather than by the
six-seed benchmark, which is the more useful lesson: a benchmark at one shape is a
benchmark of one shape.

---

## When does exact subset-sum earn its keep?

Since the ablation says "never," the honest follow-up is *never at what volume?*

```
   pay/day  credits   T0+T1     +T2   delta  +gateway  ambiguous
         7       24  35.0%  85.0%  +50.0    95.0%          3
        10       44  48.7%  84.6%  +35.9    97.4%          4
        17       53  44.7%  72.3%  +27.7    97.9%         11
        25       47  50.0%  54.8%   +4.8    95.2%         17
        37       43  50.0%  50.0%   +0.0    86.8%          9
        58       49  47.7%  47.7%   +0.0    79.5%          7
```

The solver is **decisive for a small merchant and worthless for a large one**, and the
mechanism is simple: the subsets of a capture day grow as 2ⁿ while the range of plausible
settlement amounts grows linearly. Past roughly 25 payments a day, collisions become
certain, an exact sum stops being evidence, ambiguity abstentions climb, and the API fetch
takes over.

So the pipeline is not a fixed ladder — it is a portfolio whose right mix depends on the
merchant's volume. A kirana shop needs the solver and no API budget. A large merchant
needs the API budget and barely benefits from the solver. `./run.sh scaling` reproduces it.

---

## The one number that matters

**Zero false matches**, because the engine is allowed to stop.

An unmatched credit costs an analyst five minutes. A wrongly matched credit corrupts the
ledger, misstates which orders were paid for, and is discovered at audit. Those costs are
not comparable, so Khata never trades one for the other. Every abstention is typed, owned,
and actionable:

```
NOT_A_SETTLEMENT  ×2  ₹3,69,000.00 [✓ 2/2 correct to abstain]
  Finance · Classify as direct customer receipt and post to AR, not to gateway clearing.

DUPLICATE_UTR     ×1  ₹31,617.79   [✓ 1/1 correct to abstain]
  Treasury · Confirm with the bank whether the statement line is a duplicate posting.

AMBIGUOUS_SUBSET  ×2  ₹12,000.00   [✓ 2/2 correct to abstain]
  Finance · Pull the settlement breakup report from the gateway dashboard for this UTR.
```

`correct to abstain` is scored against ground truth: it counts the exceptions where
abstaining was the *right answer*, not a failure.

---

## Quickstart

```bash
./run.sh offline      # deterministic tiers, ~0.7 s
./run.sh run          # full pipeline including the Tier 3 adjudicator
./run.sh dashboard    # http://127.0.0.1:8787
./run.sh ablate       # marginal contribution of each capability
./run.sh scaling      # subset-sum's value vs daily volume
./run.sh benchmark --no-llm --seeds 6
./run.sh test         # 65 tests
```

First run creates the venv. `ANTHROPIC_API_KEY` enables Tier 3; **without it everything
still works** — Tier 3 degrades to its deterministic narration parser and every credit it
cannot clear is logged as `NEEDS_LLM_REVIEW` rather than silently dropped. Set
`--gateway-budget 0` to see the fully offline result.

---

## How it works

Seven tiers, cheapest evidence first. A tier that cannot commit escalates rather than
guesses.

```
T0a SOURCE ──────────── is this a gateway credit at all?           free
T0  UTR lookup ──────── bank reference matches an advice           free
T1  Advice amount ───── advice net equals the credit, incl. splits free
T2  Subset-sum ──────── exact reconstruction, net of MDR + GST     free
T2b Cash + recon ────── identify the settlement, then buy back
                        its breakup with one budgeted API call     1 API call
T3  Adjudicator ─────── reads the narration; may abstain           ~2.3k tokens
T4  Group partition ─── exact cover over a day (off by default)    free
    ↓
    Exception queue ─── typed, owned, actionable
```

### Tier 2 is still the interesting one, even at +0.0

A ₹3,47,182.16 credit arrived. Which of the 23 open payments, net of 2% MDR and 18% GST on
that MDR, sum to exactly that? `khata/subsetsum.py` solves it exactly, with automatic
strategy selection:

| Strategy | Cost | Complete? |
|---|---|---|
| `prefilter` | O(n) | proves infeasibility outright |
| `full_pool` | O(n) | no |
| `complement_k` / `forward_k` | O(nᵏ), k ≤ 3 | no |
| `meet_in_middle` | O(2^(n/2)) | **yes — enumerates the whole space** |
| `bitset_dp` | n bigint shift-ors, then capped DP | feasibility only |

It is preferred wherever the pool is small enough (n ≤ 30) precisely because it is
*complete*, and completeness is what licenses everything downstream: **proving that four
distinct decompositions exist is what makes the cash-only fallback and the API fetch
legitimate rather than lazy.** Without it you would either guess or refuse everything.

### Three things the solver gets right that a naive one does not

**1. It distinguishes ambiguity from a labelling tie.** Two decompositions using different
payment *ids* but the same multiset of *amounts* are economically identical, so they are
canonicalised and the match proceeds. Genuinely different amount multisets mean the credit
has more than one valid explanation, and the engine refuses:

```
AMBIGUOUS_SUBSET  —  4 distinct payment sets reconcile to this credit exactly.
                     The amount cannot tell them apart; posting one would be a
                     guess recorded as a fact.
```

**2. It prefers the causal explanation over the coincidental one.** A chargeback raised on
the payout date is netted off that payout — that is how the gateway settles. So when one
exists, the netted search runs *first*. Searching payments-only first found a coincidence
instead: drop some unrelated payment worth exactly the chargeback and the arithmetic
balances just as well. That bug produced confident, precise, wrong answers until the
ordering was inverted.

**3. It never claims uniqueness it has not proven.** Three separate bugs in this project
were the same mistake — a uniqueness claim resting on a search that skipped something. A
truncated candidate list, a skipped capture day, and a capped enumeration all now force
the score below the posting floor instead of looking decisive.

### The gateway client is deliberately expensive

`khata/gateway.py` simulates Razorpay's settlement endpoints under a hard call budget.
Three properties keep it a data source rather than an oracle:

- **It answers "which payments", never "which settlement."** Point it at the wrong
  settlement and it returns the wrong payments with total confidence. Attribution
  precision still carries the entire load — there is a test that asserts exactly this.
- **Every call is budgeted and refused past the budget**, because pulling a quarter of
  history is the thing merchants cannot do. That is *why* this problem is solved by
  arithmetic first.
- **Every call records the credit that justified it**, so "why did we spend that call" has
  an answer in the audit trail.
- **Fetched breakups are re-verified.** The recovered nets must reconcile to the credit,
  allowing for refunds and chargebacks netted off the same cycle, or the fetch is
  discarded and the credit stays cash-only.

### Tier 3 earns its place, and is allowed to fail

The adjudicator runs last, on the residue. What reaches it is prose a human would read:

```
"NEFT RAZORPAY MERCHANT PAYOUT ORD 4471,4472,4478 AND OTHERS LESS ADJ"
```

No UTR. The amount matches no advice, because of an unexplained ₹47 gateway adjustment.
The only signal is a reference in an unknown format naming *order* ids — recovering the
settlement means joining orders to payments to the cycle they belonged to, which no regex
over that string can do:

> *All three order references in the narration (312, 306, 308) resolve to payments in
> `setl_0000000595`, whose value date matches the credit; the ₹47.00 shortfall is a
> post-advice gateway adjustment.* — confidence 0.93

Guardrails, all tested:

- **Its arithmetic is never trusted.** A proposed payment set is re-added from our own
  records; if the nets do not sum to the credit, the verdict is rejected however confident
  it sounds.
- **A settlement id it names must exist and have a breakup on file**, or the verdict is
  treated as an abstention.
- **`abstain` is a first-class decision** and the prompt prefers it. On both planted
  ambiguities the model abstained at confidence 0.3 and 0.5 — it even flagged the
  suspiciously round ₹6,000 target.
- **Confidence below 0.80 does not post.**
- **An API failure becomes an exception, never a match.**
- **A per-batch call cap** stops a runaway escalating without a budget.
- **Free wins first.** Explicit settlement references are regex-recoverable and non-gateway
  counterparties are a deterministic classification, so orphan credits never reach the
  model.

### The ledger is real double-entry

```
Payment captured   Dr gateway_clearing, mdr_expense, gst_input_credit   Cr revenue
Refund issued      Dr refunds_paid                       Cr gateway_clearing
Chargeback         Dr chargeback_losses                  Cr gateway_clearing
Credit matched     Dr bank                               Cr gateway_clearing
Credit unmatched   Dr bank                               Cr suspense
Credit explained   Dr suspense                           Cr gateway_clearing
```

`assert_balanced()` runs after every batch; if debits ever stop equalling credits the run
raises, however good the match rate looks. Unattributed cash goes to **suspense**, never
forced into clearing. The last line is a reclassification journal — posting it rather than
editing the original keeps the trail intact: the credit was unexplained, then it was
explained, and both facts remain in the ledger.

---

## Why the metrics are trustworthy

**The generator and the matchers share no code** beyond the fee model. If the matcher
reused the generator's idea of how a settlement is built, the match rate would measure
nothing but our own self-consistency. The fee model is the one deliberate exception: the
gateway and the merchant genuinely compute MDR the same way.

**Ground truth is structurally withheld.** `Batch.visible()` has no `ground_truth` key; a
settlement whose breakup is unavailable serialises `payment_ids: []` while the answer key
retains the real set; and settlements the merchant never exported are absent from
`Batch.advices` entirely while remaining in `gateway_ledger`. There are tests for each.

**Break classes are stratified, not sampled.** At this batch size an independent draw per
credit routinely produces zero instances of the rarer classes. All twelve are guaranteed:

| Break class | What makes it hard |
|---|---|
| `CLEAN_UTR` | nothing — the easy baseline |
| `ADVICE_NO_UTR` | generic narration; only the amount identifies the settlement |
| `BUNDLED_NO_ADVICE` | breakup never issued; needs subset-sum or a fetch |
| `MISSING_ADVICE` | **no local record at all** — nothing to look up, nothing to fetch |
| `REFUND_NETTED` | refunds debited mid-cycle |
| `CHARGEBACK_DEBIT` | dispute netted off the payout |
| `PARTIAL_SPLIT` | one payout, two statement lines; neither reconciles alone |
| `TIMING_SKEW` | credit lands T+5 to T+7, outside the contractual window |
| `DUPLICATE_UTR` | bank posted the same transfer twice |
| `NARRATION_ONLY` | arithmetic fails by construction; prose is the only signal |
| `ORPHAN_CREDIT` | not a gateway settlement at all |
| `AMBIGUOUS_SUBSET` | two disjoint payment sets reconcile exactly — abstaining is correct |

The last three are scored **inverted**: matching them is the failure. Zero-MDR UPI makes
`AMBIGUOUS_SUBSET` constructible exactly — `{1200,1800,3000}` and `{2500,3500}` both net
to ₹6,000, and nothing distinguishes them.

**One honest caveat, reported rather than buried.** A category called `line_equivalent`
holds credits whose payment set differs from ground truth only by swapping two zero-MDR UPI
payments of identical value (₹399 for ₹399). The amount multiset, the total and the cash
attribution are provably identical, and nothing in the data can distinguish the twins.
They are counted as cash-correct, *not* strictly line-correct, and surfaced as their own
line so the number sits in neither the wins nor the losses. `line_recall` excludes them;
`line_recall_effective` includes them.

**A known limitation, measured.** Above ~50 payments a day, cash recall falls to ~79%
because more capture days exceed the exact-search cap and are skipped. Raising the cap was
tested: it buys +4.5 points of cash recall for 5× the wall time and +0.0 line recall, so
the default stays where it is.

---

## Dashboard

`./run.sh dashboard` → KPI tiles, outcome breakdown, tier funnel, per-break-class recall,
the exception queue with owners, and the full audit trail: every credit, every tier it
passed through, and why each one declined. The **API budget** field is live — measured
straight off the running server:

| API budget | Cash recall | Line recall | Cash-only credits | Calls spent | False matches |
|---|---|---|---|---|---|
| 0 | 85.7% | 52.4% | 12 | 0 | 0 |
| 10 | 90.5% | 69.0% | 7 | 10 | 0 |
| 40 | 95.2% | 90.5% | 0 | 20 | 0 |

Precision stays at 100% the whole way down, which is the point: the budget buys *recall*,
never correctness. Note the last row spends 20 of its 40 — it asks for what it needs. Light and dark,
palette validated for colour-vision deficiency.

---

## Layout

```
khata/
  money.py          paise-only arithmetic, Indian number formatting
  fees.py           MDR + GST, rounded exactly as the gateway does
  models.py         merchant records vs gateway records vs the sealed answer key
  generator.py      synthetic batches, stratified break classes, ground truth
  subsetsum.py      exact reconstruction; five strategies, auto-selected
  groupsolve.py     exact-cover partition over a day (off by default)
  gateway.py        budgeted, audited settlement-recon client
  ledger.py         double-entry with enforced invariants
  audit.py          append-only decision log (JSONL)
  reason_codes.py   exception taxonomy: owner + next action per code
  matchers/
    tier0_source.py is this a gateway credit at all -- asked first
    tier0_utr.py    bank reference lookup; duplicate-posting detection
    tier1_advice.py advice amount match; split-payout pairing
    tier2_subset.py subset-sum with cross-day and sibling constraints
    tier2b_cash.py  settlement identification, then budgeted breakup recovery
    tier3_llm.py    the adjudicator, with a deterministic pre-pass
  engine.py         tier orchestration + the group phase
  evaluate.py       two-level scoring against held-out truth
  report.py         terminal report, ablation, scaling
  api.py            FastAPI
  cli.py            run · benchmark · ablate · scaling · generate · serve
static/index.html   dashboard
tests/              65 tests
```

Model: `claude-opus-5` via the Anthropic Python SDK, structured output through
`messages.parse` with a Pydantic verdict schema, so an unparseable answer is impossible by
construction.

---

## What I would build next

Having been wrong twice about what mattered, with less confidence than last time:

- **Real Razorpay test-mode APIs** in place of the generator, keeping the generator as the
  scoring harness — the metrics only exist because ground truth does.
- **Spend the API budget where it pays most.** Calls are currently spent first-come; the
  ablation says they are the highest-yield capability in the system, so they should be
  allocated by expected value — largest unexplained amount first, or wherever a fetch
  would collapse the most ambiguity.
- **Make the tier mix volume-aware.** The scaling table says the right configuration
  differs by an order of magnitude between a kirana shop and a large merchant. That should
  be selected from observed daily volume, not hardcoded.
- **Forward cash forecasting** from unsettled payments. The ledger already holds everything
  needed.
