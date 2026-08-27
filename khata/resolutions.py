"""Manual resolutions -- what a human decided about a credit the engine refused.

The engine's headline numbers are only worth something because nothing is
allowed to inflate them, so this store is kept strictly to one side of the
measurement. A resolution recorded here **never** re-enters the tier pipeline
and never counts toward precision or recall. It is an overlay: the dashboard
shows which exceptions a person has already cleared, and the engine's own score
stays exactly what it was.

That separation is the whole design constraint. The tempting version -- feed
resolutions back in as a Tier 5, watch the match rate climb -- would produce a
number that measures how much an analyst typed, reported as though it measured
how well the engine reconciles. The queue is a work surface; the benchmark is a
benchmark; they do not touch.

Storage is a single JSON file, keyed by (batch_id, bank_txn_id). Both are stable
for a given seed and batch configuration, so a resolution recorded today is
still attached to the same credit tomorrow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("data/resolutions.json")

# What a human is allowed to say about a credit. Deliberately small: these are
# the outcomes an accountant can actually justify to an auditor.
ACTIONS = {
    "match_settlement": "Attributed by hand to a named settlement.",
    "not_a_settlement": "Not gateway money; belongs elsewhere in the books.",
    "duplicate": "Duplicate bank posting; no cash to attribute.",
    "written_off": "Accepted as unexplained and written off.",
    "chasing": "Left open on purpose; someone is chasing it.",
}


@dataclass
class Resolution:
    batch_id: str
    bank_txn_id: str
    action: str
    note: str = ""
    settlement_id: str | None = None
    payment_ids: list[str] = field(default_factory=list)
    resolved_by: str = "unknown"
    resolved_at: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}; "
                             f"expected one of {sorted(ACTIONS)}")
        if not self.resolved_at:
            self.resolved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def key(self) -> str:
        return f"{self.batch_id}:{self.bank_txn_id}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_label"] = ACTIONS[self.action]
        return d


class ResolutionStore:
    """A JSON file of human decisions. Last write for a credit wins."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._items: dict[str, Resolution] = {}
        self.load()

    def load(self) -> None:
        self._items = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text() or "{}")
        except json.JSONDecodeError:
            # A corrupt store must not take the dashboard down with it. The file
            # is left on disk untouched so it can be inspected by hand.
            return
        for k, v in raw.items():
            try:
                self._items[k] = Resolution(**v)
            except (TypeError, ValueError):
                continue  # skip records this version cannot read

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {k: asdict(v) for k, v in self._items.items()}, indent=2, sort_keys=True))

    def put(self, r: Resolution) -> Resolution:
        self._items[r.key] = r
        self.save()
        return r

    def drop(self, batch_id: str, bank_txn_id: str) -> bool:
        removed = self._items.pop(f"{batch_id}:{bank_txn_id}", None) is not None
        if removed:
            self.save()
        return removed

    def for_batch(self, batch_id: str) -> dict[str, Resolution]:
        """Resolutions for one batch, keyed by bank_txn_id."""
        return {r.bank_txn_id: r for r in self._items.values()
                if r.batch_id == batch_id}

    def all(self) -> list[Resolution]:
        return sorted(self._items.values(), key=lambda r: r.resolved_at, reverse=True)
