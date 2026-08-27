"""Terminal report.

Ordered so the least flattering numbers are impossible to skip: the exception
queue and the false-match cost sit above the fold, not in an appendix.
"""

from __future__ import annotations

from .engine import RunResult
from .evaluate import Metrics
from .forecast import Forecast
from .money import fmt
from .reason_codes import get as get_code

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def _bar(frac: float, width: int = 24) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _rule(char: str = "─", width: int = 78) -> str:
    return DIM + char * width + RESET


def render(m: Metrics, r: RunResult, batch_id: str) -> str:
    L: list[str] = []
    add = L.append

    add("")
    add(f"{BOLD}KHATA{RESET}  settlement reconciliation  {DIM}·{RESET}  batch {batch_id}")
    add(_rule())
    add(f"  {m.payments} payments   {m.total_credits} bank credits   "
        f"{fmt(m.amount_total_paise)} credited")
    add(f"  {m.wall_ms:.0f} ms wall   {m.throughput_credits_per_s:.0f} credits/s   "
        f"{m.llm_calls} adjudicator calls ({m.llm_tokens:,} tokens) for "
        f"{m.total_credits} credits")
    add("")

    add(f"{BOLD}Headline{RESET}")
    add(f"  cash attribution   precision {_pc(m.cash_precision)}   "
        f"recall {_pc(m.cash_recall)}   {_bar(m.cash_recall)}")
    add(f"  line attribution   precision {_pc(m.line_precision)}   "
        f"recall {_pc(m.line_recall)}   {_bar(m.line_recall)}")
    add(f"  {DIM}cash = attributed to the right settlement · "
        f"line = exact payment set recovered{RESET}")
    add("")

    fp_colour = GREEN if m.false_matches == 0 else RED
    add(f"{BOLD}Where every credit landed{RESET}")
    add(f"  {GREEN}line matched{RESET}         {m.line_correct:3d}   "
        f"exact payment set recovered")
    if m.line_equivalent:
        add(f"  {GREEN}line equivalent{RESET}      {m.line_equivalent:3d}   "
            f"{DIM}same amounts, indistinguishable ids (zero-MDR UPI twins){RESET}")
    add(f"  {CYAN}cash matched{RESET}         {m.cash_only:3d}   "
        f"{DIM}right settlement, breakup not recoverable{RESET}")
    add(f"  {GREEN}correct abstention{RESET}   {m.correct_abstentions:3d}   "
        f"{DIM}should not have been matched, and was not{RESET}")
    add(f"  {YELLOW}missed{RESET}               {m.missed:3d}   "
        f"{DIM}resolvable, left open{RESET}")
    add(f"  {fp_colour}false match{RESET}          {m.false_matches:3d}   "
        f"{fp_colour}{fmt(m.amount_false_matched_paise)} wrongly attributed{RESET}")
    add("")

    add(f"{BOLD}Tier funnel{RESET}  {DIM}(cheapest first; a token is only spent at T3){RESET}")
    order = ["T0-UTR", "T1-ADVICE", "T2-SUBSET", "T2B-CASH", "T3-ADJUDICATOR"]
    for tier in order:
        t = m.per_tier.get(tier)
        if not t:
            continue
        share = t["posted"] / m.total_credits if m.total_credits else 0
        add(f"  {tier:16s} posted {t['posted']:3d}  exceptions {t['exceptions']:3d}  "
            f"precision {t['precision_pct']:5.1f}%  {_bar(share, 18)} {t['amount_display']:>16s}")
    add("")

    add(f"{BOLD}By break class{RESET}  {DIM}(stratified: every class is exercised){RESET}")
    add(f"  {'scenario':22s} {'n':>3s} {'line':>5s} {'equiv':>6s} {'cash':>5s} "
        f"{'miss':>5s} {'abst':>5s} {'FP':>3s} {'cash recall':>12s}")
    for name, s in m.per_scenario.items():
        colour = RED if s["false_match"] else (YELLOW if s["missed"] else "")
        end = RESET if colour else ""
        add(f"  {colour}{name:22s} {s['total']:3d} {s['line_matched']:5d} "
            f"{s['line_equivalent']:6d} {s['cash_matched']:5d} {s['missed']:5d} "
            f"{s['correct_abstention']:5d} {s['false_match']:3d} "
            f"{s['cash_recall_pct']:11.1f}%{end}")
    add("")

    add(f"{BOLD}Exception queue{RESET}  {DIM}(what the engine refused to touch, and who owns it){RESET}")
    if not m.exceptions:
        add(f"  {DIM}empty{RESET}")
    for e in m.exceptions:
        rc = get_code(e["code"])
        ok = e["correct_to_abstain"]
        flag = f"{GREEN}{ok}/{e['count']} correct to abstain{RESET}" if ok else \
               f"{YELLOW}0/{e['count']} correct to abstain{RESET}"
        add(f"  {BOLD}{e['code']}{RESET}  ×{e['count']}  {e['amount_display']}  [{flag}]")
        add(f"    {rc.title}")
        add(f"    {DIM}owner {rc.owner} · {rc.next_action}{RESET}")
        add(f"    {DIM}e.g. {', '.join(e['examples'])}{RESET}")
    add("")

    tb = r.ledger.trial_balance()
    status = f"{GREEN}BALANCED{RESET}" if tb["balanced"] else f"{RED}IMBALANCED{RESET}"
    add(f"{BOLD}Ledger{RESET}  {status}   {tb['postings']} postings")
    add(f"  debits {fmt(tb['total_debits_paise'])}   credits {fmt(tb['total_credits_paise'])}")
    for acct in ("bank", "gateway_clearing", "suspense", "mdr_expense",
                 "gst_input_credit", "refunds_paid", "chargeback_losses"):
        v = tb["balances"].get(acct, 0)
        if v:
            mark = f"  {YELLOW}<- unexplained cash{RESET}" if acct == "suspense" else ""
            add(f"    {acct:20s} {fmt(v):>18s}{mark}")
    add("")
    return "\n".join(L)


def _pc(x: float) -> str:
    colour = GREEN if x >= 0.95 else (YELLOW if x >= 0.80 else RED)
    return f"{colour}{x:6.1%}{RESET}"


def render_benchmark(rows: list[dict]) -> str:
    """Cross-seed table. One good seed proves nothing."""
    L = [""]
    L.append(f"{BOLD}Held-out benchmark{RESET}  {DIM}(independent seeds, "
             f"never tuned against){RESET}")
    L.append(_rule())
    L.append(f"  {'seed':>5s} {'credits':>8s} {'cash P':>8s} {'cash R':>8s} "
             f"{'line R':>8s} {'FP':>4s} {'wrong Rs':>14s} {'ms':>8s} {'LLM':>5s}")
    for r in rows:
        colour = RED if r["false_matches"] else ""
        end = RESET if colour else ""
        L.append(f"  {colour}{r['seed']:5d} {r['credits']:8d} {r['cash_precision']:7.1%} "
                 f"{r['cash_recall']:7.1%} {r['line_recall']:7.1%} {r['false_matches']:4d} "
                 f"{r['wrong_amount']:>14s} {r['wall_ms']:8.0f} {r['llm_calls']:5d}{end}")
    n = len(rows)
    L.append(_rule())
    L.append(f"  {'mean':>5s} {sum(r['credits'] for r in rows)//n:8d} "
             f"{sum(r['cash_precision'] for r in rows)/n:7.1%} "
             f"{sum(r['cash_recall'] for r in rows)/n:7.1%} "
             f"{sum(r['line_recall'] for r in rows)/n:7.1%} "
             f"{sum(r['false_matches'] for r in rows):4d} "
             f"{'':>14s} {sum(r['wall_ms'] for r in rows)/n:8.0f} "
             f"{sum(r['llm_calls'] for r in rows):5d}")
    L.append("")
    return "\n".join(L)


def render_ablation(rows: list[dict], batch_id: str) -> str:
    """What each capability actually buys, in order of addition.

    Written as a table rather than prose because the interesting entries are the
    ones that buy nothing, and prose is too easy to write around them.
    """
    L = ["", f"{BOLD}Ablation{RESET}  {DIM}· batch {batch_id} · each row adds one "
             f"capability to the row above{RESET}", _rule()]
    L.append(f"  {'configuration':34s} {'cash R':>7s} {'line R':>7s} {'d':>6s} "
             f"{'eff R':>7s} {'cash P':>7s} {'FP':>3s} {'cash-only':>10s} "
             f"{'API':>4s} {'LLM':>4s} {'ms':>7s}")
    prev_line = None
    for r in rows:
        delta = "  --  " if prev_line is None else f"{(r['line_recall']-prev_line)*100:+5.1f} "
        colour = RED if r["false_matches"] else ""
        end = RESET if colour else ""
        L.append(f"  {colour}{r['label']:34s} {r['cash_recall']:6.1%} {r['line_recall']:6.1%} "
                 f"{delta:>6s} {r['line_recall_effective']:6.1%} {r['cash_precision']:6.1%} "
                 f"{r['false_matches']:3d} {r['cash_only']:10d} {r['api_calls']:4d} "
                 f"{r['llm_calls']:4d} {r['wall_ms']:7.0f}{end}")
        prev_line = r["line_recall"]
    L.append(_rule())
    L.append(f"  {DIM}d = change in strict line recall from the row above. "
             f"eff R also counts equal-amount id ties.{RESET}")
    L.append(f"  {DIM}Rows are cumulative, so deltas are order-dependent: each tier is "
             f"measured given the ones above it.{RESET}")
    L.append("")
    return "\n".join(L)


def render_scaling(rows: list[dict]) -> str:
    """Exact subset-sum's marginal value against the merchant's daily volume.

    The mechanism is worth stating because the shape is not obvious: the number
    of candidate subsets in a capture day grows as 2^n, while the range of
    plausible settlement amounts grows only linearly. So collisions become
    certain, an exact sum stops being evidence, and the tier that carries a
    low-volume merchant does nothing at all for a high-volume one.
    """
    L = ["", f"{BOLD}When does exact subset-sum earn its keep?{RESET}", _rule()]
    L.append(f"  {'pay/day':>8s} {'credits':>8s} {'T0+T1':>7s} {'+T2':>7s} {'delta':>7s} "
             f"{'+gateway':>9s} {'ambiguous':>10s}")
    for r in rows:
        d = (r["with_t2"] - r["without_t2"]) * 100
        colour = GREEN if d >= 20 else (YELLOW if d >= 5 else DIM)
        L.append(f"  {r['per_day']:8.0f} {r['credits']:8d} {r['without_t2']:6.1%} "
                 f"{r['with_t2']:6.1%} {colour}{d:+6.1f}{RESET} {r['with_gateway']:8.1%} "
                 f"{r['ambiguous']:10d}")
    L.append(_rule())
    L.append(f"  {DIM}Subsets of a capture day grow as 2^n; plausible settlement amounts")
    L.append(f"  grow linearly. Past roughly 25 payments a day an exact sum stops being")
    L.append(f"  evidence, ambiguity abstentions rise, and the API fetch does the work")
    L.append(f"  instead. Below that, subset-sum is what carries the batch.{RESET}")
    L.append("")
    return "\n".join(L)


def render_forecast(f: Forecast, batch_id: str) -> str:
    """Forward payout schedule. Negative days are real: a cycle with more
    refunds and chargebacks than captures is a debit, and the merchant should
    see that coming rather than discover it in the statement."""
    out: list[str] = []
    out.append(f"\n{BOLD}KHATA{RESET}  cash forecast  {DIM}·{RESET}  batch {batch_id}")
    out.append(_rule())
    out.append(f"  as of {BOLD}{f.as_of}{RESET}   settlement lag T+{f.lag_days}   "
               f"{DIM}last day with complete capture data{RESET}")

    if not f.days:
        out.append(f"\n  {DIM}Nothing outstanding: every capture in this batch has "
                   f"already reached its payout date.{RESET}\n")
        return "\n".join(out)

    out.append(f"\n{BOLD}Expected payouts{RESET}")
    out.append(f"  {'payout date':<13} {'pays':>5} {'gross':>15} {'fees+GST':>13} "
               f"{'refunds':>13} {'chargebacks':>13} {'expected':>15}")
    for d in f.days:
        colour = GREEN if d.expected_paise >= 0 else RED
        out.append(
            f"  {str(d.payout_date):<13} {d.payments:>5} {fmt(d.gross_paise):>15} "
            f"{fmt(d.mdr_paise + d.gst_paise):>13} {fmt(d.refunds_paise):>13} "
            f"{fmt(d.chargebacks_paise):>13} "
            f"{colour}{fmt(d.expected_paise):>15}{RESET}")
    out.append(_rule())
    total_colour = GREEN if f.total_paise >= 0 else RED
    out.append(f"  {'total':<13} {f.payments:>5} {'':>15} {'':>13} {'':>13} {'':>13} "
               f"{BOLD}{total_colour}{fmt(f.total_paise):>15}{RESET}")
    neg = [d for d in f.days if d.expected_paise < 0]
    if neg:
        out.append(f"\n  {YELLOW}{len(neg)} cycle(s) net to a debit{RESET}  "
                   f"{DIM}more refunds and chargebacks than captures; the bank line "
                   f"will be a withdrawal, not a credit.{RESET}")
    out.append(f"\n  {DIM}Arithmetic on captured payments at T+{f.lag_days}, netted the "
               f"way the payout nets. Not a prediction: a payment captured on the "
               f"{f.as_of} cannot settle before {f.as_of.fromordinal(f.as_of.toordinal() + f.lag_days)}.{RESET}\n")
    return "\n".join(out)
