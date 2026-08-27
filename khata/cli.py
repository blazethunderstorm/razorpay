"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .engine import Engine
from .evaluate import evaluate
from .generator import generate
from .money import fmt
from .forecast import forecast
from .report import render, render_benchmark, render_forecast


def _engine(a: argparse.Namespace) -> Engine:
    return Engine(
        use_llm=not a.no_llm, lookback_days=a.lookback, model=a.model,
        confidence_floor=a.floor, subset_budget_ms=a.budget,
        max_llm_calls=a.max_llm_calls, gateway_budget=a.gateway_budget,
        gateway_policy=a.gateway_policy, live_gateway=getattr(a, "live_gateway", False),
        enable_group=getattr(a, "group", False),
        audit_path=a.audit or f"data/audit_{a.seed}.jsonl",
    )


def cmd_run(a: argparse.Namespace) -> int:
    batch = generate(seed=a.seed, n_payments=a.payments, days=a.days)
    result = _engine(a).run(batch)
    metrics = evaluate(batch, result)
    print(render(metrics, result, batch.batch_id))

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({
            "metrics": metrics.to_dict(),
            "outcomes": [o.to_dict() for o in result.outcomes],
            "ledger": result.ledger.trial_balance(),
            "llm": result.llm, "config": result.config,
        }, indent=2, default=str))
        print(f"  wrote {a.json}")
    if result.audit.path:
        print(f"  audit trail: {result.audit.path} "
              f"({len(result.audit.records)} decisions)\n")
    return 0 if metrics.false_matches == 0 else 1


def cmd_benchmark(a: argparse.Namespace) -> int:
    rows = []
    for i in range(a.seeds):
        seed = a.seed + i * 101
        batch = generate(seed=seed, n_payments=a.payments, days=a.days)
        ns = argparse.Namespace(**{**vars(a), "seed": seed})
        result = _engine(ns).run(batch)
        m = evaluate(batch, result)
        rows.append({
            "seed": seed, "credits": m.total_credits,
            "cash_precision": m.cash_precision, "cash_recall": m.cash_recall,
            "line_recall": m.line_recall, "false_matches": m.false_matches,
            "wrong_amount": fmt(m.amount_false_matched_paise),
            "wall_ms": m.wall_ms, "llm_calls": m.llm_calls,
        })
        print(f"  seed {seed}: cash P {m.cash_precision:.1%} R {m.cash_recall:.1%}, "
              f"line R {m.line_recall:.1%}, FP {m.false_matches}", flush=True)
    print(render_benchmark(rows))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
    return 0 if all(r["false_matches"] == 0 for r in rows) else 1


# T0-SOURCE is in every configuration: classifying non-gateway credits is a
# precondition for matching, not an optional capability. Ablating it away does not
# measure a tier's value, it just lets subset-sum invent settlements.
ABLATION = [
    ("T0 only  (UTR lookup)",            {"T0-SOURCE", "T0-UTR"},                                  0, False),
    ("+T1      (advice amount)",         {"T0-SOURCE", "T0-UTR", "T1-ADVICE"},                     0, False),
    ("+T2      (subset-sum)",            {"T0-SOURCE", "T0-UTR", "T1-ADVICE", "T2-SUBSET"},        0, False),
    ("+T2b     (cash attribution)",      {"T0-SOURCE", "T0-UTR", "T1-ADVICE", "T2-SUBSET", "T2B-CASH"}, 0, False),
    ("+gateway (recon fetch)",           {"T0-SOURCE", "T0-UTR", "T1-ADVICE", "T2-SUBSET", "T2B-CASH"}, 40, False),
    ("+T4      (group partition)",       {"T0-SOURCE", "T0-UTR", "T1-ADVICE", "T2-SUBSET", "T2B-CASH"}, 40, True),
    ("+T3      (adjudicator)",           None,                                        40, False),
]


def cmd_ablate(a: argparse.Namespace) -> int:
    """Marginal contribution of each capability, measured rather than asserted."""
    from .report import render_ablation
    batch = generate(seed=a.seed, n_payments=a.payments, days=a.days)
    rows = []
    for label, tiers, gw, group in ABLATION:
        use_llm = tiers is None and not a.no_llm
        result = Engine(use_llm=use_llm, enabled_tiers=frozenset(tiers) if tiers else None,
                        gateway_budget=gw, enable_group=group,
                        lookback_days=a.lookback, model=a.model,
                        subset_budget_ms=a.budget).run(batch)
        m = evaluate(batch, result)
        rows.append({
            "label": label, "cash_recall": m.cash_recall, "line_recall": m.line_recall,
            "line_recall_effective": m.line_recall_effective,
            "cash_precision": m.cash_precision, "false_matches": m.false_matches,
            "wrong": fmt(m.amount_false_matched_paise), "cash_only": m.cash_only,
            "open": m.missed + m.correct_abstentions, "wall_ms": m.wall_ms,
            "llm_calls": m.llm_calls, "api_calls": result.gateway["spent"],
            "group_upgrades": result.group["credits_upgraded"],
        })
        print(f"  {label:32s} cash R {m.cash_recall:6.1%}  line R {m.line_recall:6.1%}  "
              f"FP {m.false_matches}", flush=True)
    print(render_ablation(rows, batch.batch_id))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
    return 0


def cmd_scaling(a: argparse.Namespace) -> int:
    """How the value of exact subset-sum varies with the merchant's daily volume.

    The pipeline is not a fixed ladder -- which tier earns its keep depends on
    how many payments a day holds, so it is worth measuring rather than assuming.
    """
    from .report import render_scaling
    base = frozenset({"T0-SOURCE", "T0-UTR", "T1-ADVICE"})
    with_t2 = frozenset({"T0-SOURCE", "T0-UTR", "T1-ADVICE", "T2-SUBSET"})
    rows = []
    for n in (150, 250, 400, 600, 900, 1400):
        batch = generate(seed=a.seed, n_payments=n, days=a.days)
        without = evaluate(batch, Engine(use_llm=False, enabled_tiers=base,
                                         gateway_budget=0).run(batch))
        run = Engine(use_llm=False, enabled_tiers=with_t2, gateway_budget=0).run(batch)
        withs = evaluate(batch, run)
        full = evaluate(batch, Engine(use_llm=False, gateway_budget=40).run(batch))
        rows.append({
            "per_day": len(batch.payments) / a.days, "credits": len(batch.bank_txns),
            "without_t2": without.line_recall_effective,
            "with_t2": withs.line_recall_effective,
            "with_gateway": full.line_recall_effective,
            "ambiguous": sum(1 for o in run.outcomes
                             if o.reason_code == "AMBIGUOUS_SUBSET"),
        })
        print(f"  {len(batch.payments):5d} payments  "
              f"({rows[-1]['per_day']:.0f}/day)  subset-sum delta "
              f"{(rows[-1]['with_t2'] - rows[-1]['without_t2']) * 100:+5.1f}", flush=True)
    print(render_scaling(rows))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
    return 0


def cmd_generate(a: argparse.Namespace) -> int:
    batch = generate(seed=a.seed, n_payments=a.payments, days=a.days)
    out = Path(a.out or f"data/batch_{a.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(batch.to_dict(), indent=2, default=str))
    print(f"{len(batch.payments)} payments, {len(batch.bank_txns)} bank credits "
          f"-> {out}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn
    print(f"  dashboard  http://{a.host}:{a.port}/")
    uvicorn.run("khata.api:app", host=a.host, port=a.port, reload=a.reload)
    return 0


def cmd_forecast(a: argparse.Namespace) -> int:
    batch = generate(seed=a.seed, n_payments=a.payments, days=a.days)
    as_of = date.fromisoformat(a.as_of) if a.as_of else None
    f = forecast(batch, lag_days=a.lag, as_of=as_of, horizon=a.horizon)
    print(render_forecast(f, batch.batch_id))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(f.to_dict(), indent=2, default=str))
        print(f"  wrote {a.json}")
    return 0


def cmd_razorpay_check(a: argparse.Namespace) -> int:
    """Prove the Razorpay credentials work before a run depends on them."""
    from .razorpay_client import RazorpayAuthError, RazorpaySettlements
    try:
        gw = RazorpaySettlements(ledger=[], budget=2)
        h = gw.health()
    except RazorpayAuthError as e:
        print(f"  razorpay: {e}")
        return 1
    if not h["ok"]:
        print(f"  razorpay: {h['detail']}")
        for c in gw.calls:
            print(f"    {c.op} {c.argument}: {c.note}")
        return 1
    print(f"  razorpay: OK  mode={h['mode']}  "
          f"settlements visible={h['settlements_visible']}")
    if h["mode"] == "live":
        print("  warning: these are LIVE keys. Use test-mode keys (rzp_test_...).")
    gw.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="khata",
        description="Settlement reconciliation with measured, held-out accuracy.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--payments", type=int, default=600)
        sp.add_argument("--days", type=int, default=21)
        sp.add_argument("--no-llm", action="store_true",
                        help="skip the Tier 3 adjudicator entirely")
        sp.add_argument("--model", default=None, help="override KHATA_MODEL")
        sp.add_argument("--lookback", type=int, default=8)
        sp.add_argument("--floor", type=float, default=0.80,
                        help="confidence below which a match is not posted")
        sp.add_argument("--budget", type=int, default=2000,
                        help="per-credit subset-search budget in ms")
        sp.add_argument("--max-llm-calls", type=int, default=40)
        sp.add_argument("--live-gateway", action="store_true",
                        help="use the real Razorpay settlement endpoints instead of the "
                             "simulator (needs RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET)")
        sp.add_argument("--gateway-policy", choices=("fifo", "value"), default="fifo",
                        help="how to allocate the gateway call budget: fifo "
                             "(first credit to need one) or value (largest credits first)")
        sp.add_argument("--gateway-budget", type=int, default=40,
                        help="max settlement-recon API calls per batch (0 disables)")
        sp.add_argument("--group", action="store_true",
                        help="enable the exact-cover group pass (off by default: it has "
                             "never proved a unique partition -- see README)")
        sp.add_argument("--audit", default=None)
        sp.add_argument("--json", default=None)

    r = sub.add_parser("run", help="reconcile one batch and print the report")
    common(r); r.set_defaults(fn=cmd_run)

    b = sub.add_parser("benchmark", help="repeat across independent seeds")
    common(b); b.add_argument("--seeds", type=int, default=5)
    b.set_defaults(fn=cmd_benchmark)

    ab = sub.add_parser("ablate", help="marginal contribution of each tier")
    common(ab); ab.set_defaults(fn=cmd_ablate)

    sc = sub.add_parser("scaling", help="subset-sum's value vs daily volume")
    common(sc); sc.set_defaults(fn=cmd_scaling)

    g = sub.add_parser("generate", help="write a synthetic batch to disk")
    common(g); g.add_argument("--out", default=None)
    g.set_defaults(fn=cmd_generate)

    fc = sub.add_parser("forecast", help="expected payouts still to come")
    common(fc)
    fc.add_argument("--horizon", type=int, default=7, help="days ahead to show")
    fc.add_argument("--lag", type=int, default=2, help="settlement lag in days (T+N)")
    fc.add_argument("--as-of", default=None,
                    help="YYYY-MM-DD to forecast from (default: last capture date)")
    fc.set_defaults(fn=cmd_forecast)

    rz = sub.add_parser("razorpay-check", help="verify Razorpay API credentials")
    rz.set_defaults(fn=cmd_razorpay_check)

    s = sub.add_parser("serve", help="run the dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
