"""Live Razorpay settlement endpoints, behind the same budget as the simulator.

This is a drop-in for :class:`khata.gateway.GatewayClient`. It keeps the budget
accounting, the call log and the "identify the settlement yourself first"
contract exactly as they are, and swaps only where the data comes from.

Two shapes in the real API differ from the simulator and are worth knowing:

**Recon is fetched by day, not by settlement.** ``/v1/settlements/recon/combined``
returns every payment settled on a given day, across all settlements that day.
So one HTTP call can answer several ``fetch_recon`` questions. The day is cached
and charged once, which makes the live client *cheaper* per settlement than the
simulator rather than more expensive -- the budget still bounds the HTTP calls,
which is the resource that is actually scarce.

**Amounts are already in paise**, and settlement ids come back as ``setl_...``
rather than the generator's ``SETL_...``. Nothing else needs translating.

Credentials come from ``RAZORPAY_KEY_ID`` / ``RAZORPAY_KEY_SECRET``. Use test
mode keys: this client only ever issues GETs, but test mode is the right place
to point a reconciler you are still measuring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import httpx

from .gateway import GatewayCall, GatewayClient

API_ROOT = "https://api.razorpay.com/v1"
TIMEOUT_S = 30.0


class RazorpayAuthError(RuntimeError):
    """Credentials are missing or rejected."""


def credentials() -> tuple[str, str]:
    key = os.environ.get("RAZORPAY_KEY_ID", "").strip()
    secret = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()
    if not key or not secret:
        raise RazorpayAuthError(
            "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set. Get test-mode "
            "keys from the Razorpay dashboard: Settings -> API Keys -> Test Mode, "
            "then put them in .env alongside ANTHROPIC_API_KEY.")
    return key, secret


def _epoch(d: date, end: bool = False) -> int:
    t = time(23, 59, 59) if end else time(0, 0, 0)
    return int(datetime.combine(d, t, tzinfo=timezone.utc).timestamp())


@dataclass
class RazorpaySettlements(GatewayClient):
    """GatewayClient backed by the real settlement endpoints."""

    client: httpx.Client | None = None
    _recon_by_day: dict[date, dict[str, list[str]]] = field(
        default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.client is None:
            key, secret = credentials()
            self.client = httpx.Client(base_url=API_ROOT, auth=(key, secret),
                                       timeout=TIMEOUT_S,
                                       headers={"User-Agent": "khata/1.0"})

    # ---- transport ----

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """One HTTP GET. Network and auth failures are outcomes, not crashes:
        the engine must be able to carry on with whatever it already had."""
        assert self.client is not None
        try:
            r = self.client.get(path, params=params)
        except httpx.HTTPError as e:
            self.calls.append(GatewayCall("http_error", path, "-", False, str(e)))
            return None
        if r.status_code in (401, 403):
            raise RazorpayAuthError(
                f"Razorpay rejected the credentials ({r.status_code}). Check "
                f"RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET.")
        if r.status_code >= 400:
            self.calls.append(GatewayCall("http_error", path, "-", False,
                                          f"HTTP {r.status_code}: {r.text[:200]}"))
            return None
        return r.json()

    # ---- endpoints ----

    def fetch_recon(self, settlement_id: str, on_behalf_of: str) -> list[str] | None:
        """Payment ids making up one settlement.

        The recon report is per-day, so the day is fetched once and reused. A
        settlement whose day is already cached costs nothing further -- the
        budget bounds HTTP calls, which is the real constraint.
        """
        if self.hold:
            from .gateway import HeldRequest
            self.held.append(HeldRequest(settlement_id, on_behalf_of))
            return None

        day = self._day_of(settlement_id)
        if day is None:
            self._charge("fetch_recon", settlement_id, on_behalf_of, False,
                         "settlement date unknown; identify it first")
            return None

        if day not in self._recon_by_day:
            if self.remaining <= 0:
                self._charge("fetch_recon", settlement_id, on_behalf_of, False)
                return None
            payload = self._get("/settlements/recon/combined",
                                {"year": day.year, "month": day.month,
                                 "day": day.day, "count": 1000})
            if payload is None:
                return None
            by_settlement: dict[str, list[str]] = {}
            for item in payload.get("items", []):
                sid = item.get("settlement_id")
                # Only payments make up the gross; refunds and adjustments are
                # netted off it and are not part of the payment set.
                if sid and item.get("type") == "payment" and item.get("entity_id"):
                    by_settlement.setdefault(sid, []).append(item["entity_id"])
            self._recon_by_day[day] = by_settlement
            self._charge("fetch_recon", day.isoformat(), on_behalf_of, True,
                         f"{len(by_settlement)} settlements on this day")

        pids = self._recon_by_day[day].get(settlement_id)
        if not pids:
            return None
        return list(pids)

    def list_settlements(self, on_date: date, on_behalf_of: str) -> list[dict[str, Any]]:
        if self.remaining <= 0:
            self._charge("list_settlements", on_date.isoformat(), on_behalf_of, False)
            return []
        payload = self._get("/settlements", {"from": _epoch(on_date),
                                             "to": _epoch(on_date, end=True),
                                             "count": 100})
        if payload is None:
            return []
        out = []
        for item in payload.get("items", []):
            created = item.get("created_at")
            out.append({
                "settlement_id": item.get("id"),
                "net_paise": item.get("amount"),
                "utr": item.get("utr"),
                "settled_at": (datetime.fromtimestamp(created, tz=timezone.utc)
                               if created else None),
            })
        self._charge("list_settlements", on_date.isoformat(), on_behalf_of, True,
                     f"{len(out)} settlements")
        return out

    # ---- helpers ----

    def _day_of(self, settlement_id: str) -> date | None:
        """The settlement's date, from whatever the engine already knows.

        Deliberately does not go and look it up: this client resolves *which
        payments*, never *which settlement*, and a lookup here would quietly
        turn it into an oracle.
        """
        a = self._by_id.get(settlement_id)
        return a.settled_at.date() if a is not None else None

    def health(self) -> dict[str, Any]:
        """One cheap authenticated call, to prove the credentials work."""
        payload = self._get("/settlements", {"count": 1})
        if payload is None:
            return {"ok": False, "detail": "request failed; see the call log"}
        return {"ok": True, "settlements_visible": payload.get("count", 0),
                "mode": "test" if os.environ.get("RAZORPAY_KEY_ID", "")
                        .startswith("rzp_test") else "live"}

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
