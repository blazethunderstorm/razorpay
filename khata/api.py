"""HTTP surface for the dashboard.

Runs are cached by their full parameter tuple. A reconciliation is
deterministic given (seed, payments, days, llm) -- same inputs, same postings,
same audit trail -- so caching is safe and makes the dashboard feel instant
after the first load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Engine
from .evaluate import evaluate
from .generator import generate
from .money import fmt
from .forecast import forecast
from .reason_codes import all_codes
from .resolutions import ACTIONS, Resolution, ResolutionStore

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Khata", version="1.0.0",
              description="Settlement reconciliation with measured, held-out accuracy.")

_cache: dict[tuple, dict[str, Any]] = {}


def _run(seed: int, payments: int, days: int, llm: bool,
         gateway: int = 40, group: bool = False) -> dict[str, Any]:
    key = (seed, payments, days, llm, gateway, group)
    if key in _cache:
        return _cache[key]

    batch = generate(seed=seed, n_payments=payments, days=days)
    result = Engine(use_llm=llm, gateway_budget=gateway, enable_group=group,
                    audit_path=f"data/audit_{seed}.jsonl").run(batch)
    metrics = evaluate(batch, result)
    truth = {g.bank_txn_id: g for g in batch.ground_truth}
    verdict = {v.bank_txn_id: v for v in metrics.verdicts}

    credits = []
    for o in result.outcomes:
        v = verdict[o.bank_txn_id]
        g = truth[o.bank_txn_id]
        credits.append({
            **o.to_dict(),
            "label": v.label,
            "scenario": v.scenario,
            "resolvable": g.resolvable,
            "expected_count": len(v.expected),
            "predicted_count": len(v.predicted),
            "truth_note": g.note,
        })

    payload = {
        "batch": {
            "id": batch.batch_id, "seed": seed, "payments": len(batch.payments),
            "refunds": len(batch.refunds), "chargebacks": len(batch.chargebacks),
            "advices": len(batch.advices),
            "gateway_settlements": len(batch.gateway_ledger),
            "advices_with_breakup": sum(a.breakup_available for a in batch.advices),
            "credits": len(batch.bank_txns),
            "credited_display": fmt(sum(t.amount_paise for t in batch.bank_txns)),
        },
        "metrics": metrics.to_dict(),
        "credits": credits,
        "ledger": result.ledger.trial_balance(),
        "ledger_display": {k: fmt(v) for k, v in
                           result.ledger.trial_balance()["balances"].items()},
        "llm": result.llm,
        "gateway": result.gateway,
        "group": result.group,
        "config": result.config,
        "audit": result.audit.to_list(),
        "reason_codes": [c.__dict__ for c in all_codes()],
        "money": {
            "total": fmt(metrics.amount_total_paise),
            "cash_correct": fmt(metrics.amount_cash_correct_paise),
            "line_correct": fmt(metrics.amount_line_correct_paise),
            "false_matched": fmt(metrics.amount_false_matched_paise),
            "suspense": fmt(metrics.amount_suspense_paise),
        },
    }
    _cache[key] = payload
    return payload


@app.get("/api/run")
def api_run(seed: int = Query(42), payments: int = Query(600),
            days: int = Query(21), llm: bool = Query(False),
            gateway: int = Query(40), group: bool = Query(False)) -> JSONResponse:
    return JSONResponse(_run(seed, payments, days, llm, gateway, group))


@app.get("/api/credit/{bank_txn_id}")
def api_credit(bank_txn_id: str, seed: int = Query(42), payments: int = Query(600),
               days: int = Query(21), llm: bool = Query(False),
               gateway: int = Query(40), group: bool = Query(False)) -> JSONResponse:
    data = _run(seed, payments, days, llm, gateway, group)
    credit = next((c for c in data["credits"] if c["bank_txn_id"] == bank_txn_id), None)
    if credit is None:
        return JSONResponse({"error": "unknown bank_txn_id"}, status_code=404)
    return JSONResponse({
        "credit": credit,
        "audit": [a for a in data["audit"] if a["bank_txn_id"] == bank_txn_id],
    })


# One store for the process. Resolutions are an overlay on the exception queue
# and never feed the engine -- see khata/resolutions.py for why that matters.
_store = ResolutionStore()


@app.get("/api/forecast")
def api_forecast(seed: int = Query(42), payments: int = Query(600),
                 days: int = Query(21), horizon: int = Query(7),
                 lag: int = Query(2)) -> JSONResponse:
    batch = generate(seed=seed, n_payments=payments, days=days)
    return JSONResponse(forecast(batch, lag_days=lag, horizon=horizon).to_dict())


@app.get("/api/resolutions")
def api_resolutions(seed: int = Query(42)) -> JSONResponse:
    batch_id = f"batch_{seed}"
    return JSONResponse({
        "batch_id": batch_id,
        "actions": ACTIONS,
        "resolutions": {k: r.to_dict()
                        for k, r in _store.for_batch(batch_id).items()},
    })


@app.post("/api/resolve")
def api_resolve(body: dict[str, Any] = Body(...)) -> JSONResponse:
    """Record one human decision about one credit.

    Deliberately does not touch the cached run: the metrics a reviewer is
    looking at must not move because a reviewer cleared something by hand.
    """
    seed = int(body.get("seed", 42))
    bank_txn_id = str(body.get("bank_txn_id", "")).strip()
    if not bank_txn_id:
        raise HTTPException(status_code=400, detail="bank_txn_id is required")
    try:
        r = Resolution(
            batch_id=f"batch_{seed}",
            bank_txn_id=bank_txn_id,
            action=str(body.get("action", "")),
            note=str(body.get("note", "")).strip(),
            settlement_id=body.get("settlement_id") or None,
            payment_ids=list(body.get("payment_ids") or []),
            resolved_by=str(body.get("resolved_by") or "dashboard").strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(_store.put(r).to_dict())


@app.delete("/api/resolve/{bank_txn_id}")
def api_unresolve(bank_txn_id: str, seed: int = Query(42)) -> JSONResponse:
    dropped = _store.drop(f"batch_{seed}", bank_txn_id)
    if not dropped:
        raise HTTPException(status_code=404, detail="no resolution for that credit")
    return JSONResponse({"dropped": bank_txn_id})


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "cached_runs": len(_cache)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
