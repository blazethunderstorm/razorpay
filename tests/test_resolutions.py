"""Manual resolutions.

The property that matters most is negative: recording a human decision must not
move a single number in the engine's own score.
"""
import json

import pytest

from khata.engine import Engine
from khata.evaluate import evaluate
from khata.generator import generate
from khata.resolutions import ACTIONS, Resolution, ResolutionStore


def test_round_trips_through_disk(tmp_path):
    p = tmp_path / "res.json"
    s = ResolutionStore(p)
    s.put(Resolution(batch_id="batch_42", bank_txn_id="bnk_1",
                     action="not_a_settlement", note="direct NEFT",
                     resolved_by="anirudh"))
    assert ResolutionStore(p).for_batch("batch_42")["bnk_1"].note == "direct NEFT"


def test_rejects_an_action_nobody_can_justify():
    with pytest.raises(ValueError):
        Resolution(batch_id="b", bank_txn_id="x", action="looks_fine_to_me")


def test_last_write_wins(tmp_path):
    p = tmp_path / "res.json"
    s = ResolutionStore(p)
    s.put(Resolution(batch_id="b", bank_txn_id="x", action="chasing"))
    s.put(Resolution(batch_id="b", bank_txn_id="x", action="written_off"))
    assert s.for_batch("b")["x"].action == "written_off"
    assert len(s.all()) == 1


def test_drop_removes_only_that_credit(tmp_path):
    p = tmp_path / "res.json"
    s = ResolutionStore(p)
    s.put(Resolution(batch_id="b", bank_txn_id="x", action="chasing"))
    s.put(Resolution(batch_id="b", bank_txn_id="y", action="chasing"))
    assert s.drop("b", "x") is True
    assert s.drop("b", "x") is False
    assert list(s.for_batch("b")) == ["y"]


def test_batches_are_isolated(tmp_path):
    p = tmp_path / "res.json"
    s = ResolutionStore(p)
    s.put(Resolution(batch_id="batch_42", bank_txn_id="x", action="chasing"))
    s.put(Resolution(batch_id="batch_143", bank_txn_id="x", action="chasing"))
    assert list(s.for_batch("batch_42")) == ["x"]
    assert len(s.all()) == 2


def test_a_corrupt_store_does_not_take_the_dashboard_down(tmp_path):
    p = tmp_path / "res.json"
    p.write_text("{not json at all")
    assert ResolutionStore(p).all() == []
    assert p.read_text() == "{not json at all", "the bad file is left for inspection"


def test_unreadable_records_are_skipped_not_fatal(tmp_path):
    p = tmp_path / "res.json"
    p.write_text(json.dumps({
        "b:good": {"batch_id": "b", "bank_txn_id": "good", "action": "chasing"},
        "b:bad": {"batch_id": "b", "bank_txn_id": "bad", "action": "from_the_future"},
    }))
    assert list(ResolutionStore(p).for_batch("b")) == ["good"]


# Wall time and throughput differ run to run; everything else must not.
TIMING = {"wall_ms", "throughput_credits_per_s"}


def _score(metrics):
    return {k: v for k, v in metrics.to_dict().items() if k not in TIMING}


def test_resolutions_never_touch_the_score(tmp_path):
    """The whole design constraint, asserted."""
    batch = generate(seed=42, n_payments=600, days=21)
    before = _score(evaluate(batch, Engine(use_llm=False, audit_path=None).run(batch)))

    store = ResolutionStore(tmp_path / "res.json")
    cleared = 0
    for o in Engine(use_llm=False, audit_path=None).run(batch).outcomes:
        if o.outcome != "matched":
            store.put(Resolution(batch_id=batch.batch_id, bank_txn_id=o.bank_txn_id,
                                 action="written_off", note="cleared by hand"))
            cleared += 1
    assert cleared, "there should be exceptions to clear"

    after = _score(evaluate(batch, Engine(use_llm=False, audit_path=None).run(batch)))
    assert after == before


def test_every_action_carries_an_explanation():
    assert all(v.strip().endswith(".") for v in ACTIONS.values())
