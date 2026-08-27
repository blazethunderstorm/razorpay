"""HTTP surface."""
import pytest
from fastapi.testclient import TestClient

from khata.api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def run():
    r = client.get("/api/run", params={"seed": 42, "payments": 300,
                                       "days": 10, "llm": False})
    assert r.status_code == 200
    return r.json()


def test_health():
    assert client.get("/api/health").json()["ok"] is True


def test_dashboard_is_served():
    r = client.get("/")
    assert r.status_code == 200 and b"Khata" in r.content


def test_run_shape(run):
    for key in ("batch", "metrics", "credits", "ledger", "llm", "audit",
                "reason_codes", "money"):
        assert key in run
    assert run["ledger"]["balanced"] is True
    assert len(run["credits"]) == run["batch"]["credits"]


def test_every_credit_carries_its_trail(run):
    for c in run["credits"]:
        assert c["trail"], c["bank_txn_id"]
        assert c["label"] in ("line_matched", "line_equivalent", "cash_matched",
                              "correct_abstention", "missed", "false_match")


def test_credit_detail_endpoint(run):
    cid = run["credits"][0]["bank_txn_id"]
    r = client.get(f"/api/credit/{cid}", params={"seed": 42, "payments": 300,
                                                 "days": 10, "llm": False})
    assert r.status_code == 200
    assert r.json()["credit"]["bank_txn_id"] == cid
    assert r.json()["audit"]


def test_unknown_credit_is_404():
    r = client.get("/api/credit/bnk_nope", params={"seed": 42, "payments": 300,
                                                   "days": 10, "llm": False})
    assert r.status_code == 404


def test_runs_are_reproducible(run):
    again = client.get("/api/run", params={"seed": 42, "payments": 300,
                                          "days": 10, "llm": False}).json()
    assert again["metrics"]["cash_precision"] == run["metrics"]["cash_precision"]
    assert [c["payment_ids"] for c in again["credits"]] == \
           [c["payment_ids"] for c in run["credits"]]
