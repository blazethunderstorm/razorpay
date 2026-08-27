"""Append-only decision log.

Every rupee the engine moves is traceable to the tier that moved it, the
evidence it relied on, and how long it took. If a number on the dashboard
cannot be traced back to a row in here, the number does not get shown.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditRecord:
    seq: int
    at: str
    bank_txn_id: str
    amount_paise: int
    tier: str
    outcome: str              # matched | escalated | exception
    reason_code: str | None
    confidence: float
    payment_ids: list[str]
    settlement_id: str | None
    residual_paise: int
    strategy: str
    elapsed_ms: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditTrail:
    def __init__(self, path: str | Path | None = None):
        self.records: list[AuditRecord] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def log(self, **kw: Any) -> AuditRecord:
        rec = AuditRecord(
            seq=len(self.records) + 1,
            at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **kw,
        )
        self.records.append(rec)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec.to_dict()) + "\n")
        return rec

    def for_txn(self, bank_txn_id: str) -> list[AuditRecord]:
        return [r for r in self.records if r.bank_txn_id == bank_txn_id]

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]
