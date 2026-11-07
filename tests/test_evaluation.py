"""Phase 5 metric tests (fake NLI model, no downloads)."""

from types import SimpleNamespace

from ragbench.evaluation import (
    NLIScorer,
    answer_metrics,
    retrieval_metrics,
    split_claims,
    token_f1,
)


def test_split_claims_attaches_trailing_markers_and_drops_fragments():
    claims = split_claims("Python uses **UTF-8** by default. [2]\n\n| a | b |\n|---|---|\nYes.")
    assert [c.text for c in claims] == ["Python uses UTF-8 by default."]
    assert claims[0].cited == [2]
    claims = split_claims("It is served at /docs 【1†L1-L3】【2】. Use ReDoc at /redoc too [3].")
    assert [c.cited for c in claims] == [[1, 2], [3]]
    claims = split_claims("Create it with venv [1].\n```bash\npython -m venv my env here\n```")
    assert [c.text for c in claims] == ["Create it with venv ."]


def test_token_f1():
    assert token_f1("The UTF-8 encoding", "UTF-8 encoding") == 1.0
    assert token_f1("nothing shared", "UTF-8") == 0.0


def test_retrieval_metrics_uses_evidence_excerpts():
    q = {"evidence": [{"supporting_excerpt": "source files are treated as encoded in UTF-8"}]}
    text = {
        "a": "unrelated text about loops and lists in python",
        "b": "By default, Python source files are treated as encoded in UTF-8. In that",
    }
    m = retrieval_metrics(q, [{"chunk_id": "a"}, {"chunk_id": "b"}], text)
    assert m["context_recall"] == 1.0 and m["evidence_fully_retrieved"] == 1.0
    assert m["context_precision"] == 0.5 and m["relevant_ranks"] == [2]
    assert not m["hit_at_1"] and m["hit_at_3"] and m["average_precision"] == 0.5


class FakeCE:
    """Entails when every word of the hypothesis appears in the premise."""

    config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

    def predict(self, pairs, **kw):
        out = []
        for prem, hyp in pairs:
            ok = set(hyp.lower().strip(".").split()) <= set(prem.lower().replace(".", "").split())
            out.append([0.0, 1.0, 0.0] if ok else [0.0, 0.0, 1.0])
        return out


def test_answer_metrics_faithfulness_and_citations():
    nli = NLIScorer(model=FakeCE())
    contexts = ["python files use utf-8 by default", "docker images are layered"]
    ans = "Python files use utf-8 by default [2]. Python files are fast and great [1]."
    m = answer_metrics(ans, "Python files use utf-8 by default.", contexts, nli)
    assert m["claims"] == 2 and m["faithfulness"] == 0.5
    assert m["citation_support"] == 0.0  # claim 1 cites the wrong passage; claim 2 unsupported
    assert m["fact_coverage"] == 1.0
    assert m["unsupported_claims"] == ["Python files are fast and great ."]
