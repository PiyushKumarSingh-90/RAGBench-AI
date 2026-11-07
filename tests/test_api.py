"""API tests against the real indexes with the offline extractive generator."""

import pytest
from fastapi.testclient import TestClient

from api.main import app, service

pytestmark = pytest.mark.skipif(
    not (service.index_root / "dense_fixed" / "chunks.jsonl").exists(),
    reason="indexes not built (scripts/build_index.py)",
)
client = TestClient(app)


def test_metadata_endpoints(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert client.get("/health").json() == {"status": "ok"}
    assert set(client.get("/strategies").json()["strategies"]) == {
        "dense_fixed",
        "dense_semantic",
        "hybrid",
        "rerank",
    }
    gens = client.get("/generators").json()["available"]
    assert gens["extractive"] and not gens["groq"]


def test_query_returns_answer_context_and_timings():
    r = client.post(
        "/query",
        json={
            "question": "What encoding does Python assume for source files?",
            "strategy": "hybrid",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "hybrid" and len(body["retrieved"]) == 5
    assert all(x["text"] for x in body["retrieved"])
    assert {"retrieval_total", "generation", "total"} <= set(body["timings_ms"])


def test_llm_without_key_and_bad_input(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/query", json={"question": "hello there", "generator": "groq"})
    assert r.status_code == 400 and "GROQ_API_KEY" in r.json()["detail"]
    assert client.post("/query", json={"question": "hi", "strategy": "nope"}).status_code == 422


def test_saved_results_and_evaluation():
    rows = client.get("/benchmark/results/hybrid").json()
    assert len(rows) == 50 and rows[0]["evaluation"] is not None
    assert "hybrid" in client.get("/evaluation").json()["strategies"]
    assert len(client.get("/benchmark/questions").json()) == 50
