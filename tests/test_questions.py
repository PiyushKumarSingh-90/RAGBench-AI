"""Validate benchmark question datasets that exist in evaluation/datasets."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "evaluation" / "datasets"
SOURCE_FILES = sorted(DATASETS.glob("*_questions.jsonl"))
SOURCE_FILES = [p for p in SOURCE_FILES if p.name != "benchmark_questions.jsonl"]

_spec = importlib.util.spec_from_file_location(
    "validate_questions", ROOT / "scripts" / "validate_questions.py"
)
vq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vq)


@pytest.fixture(scope="module")
def corpus():
    return vq.load_corpus()


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_source_dataset_valid(path, corpus):
    records, errs = vq.load_records(path)
    errs += vq.validate_dataset(records, corpus, combined=False)
    assert errs == []


SOURCE_ORDER = ["python", "scikit_learn", "docker", "fastapi", "aws"]
COMBINED = DATASETS / "benchmark_questions.jsonl"


@pytest.mark.skipif(not COMBINED.exists(), reason="combined benchmark not created yet")
def test_combined_benchmark_valid(corpus):
    records, errs = vq.load_records(COMBINED)
    errs += vq.validate_dataset(records, corpus, combined=True)
    assert errs == []


@pytest.mark.skipif(not COMBINED.exists(), reason="combined benchmark not created yet")
def test_combined_matches_source_files():
    parts = [(DATASETS / f"{s}_questions.jsonl").read_text(encoding="utf-8") for s in SOURCE_ORDER]
    assert COMBINED.read_text(encoding="utf-8") == "".join(parts)


def test_validator_rejects_fabricated_evidence(corpus):
    path = DATASETS / "python_questions.jsonl"
    if not path.exists():
        pytest.skip("python dataset not present")
    records, _ = vq.load_records(path)
    bad = json.loads(json.dumps(records))
    bad[0]["evidence"][0]["supporting_excerpt"] = "This sentence is not in the corpus."
    bad[1]["evidence"][0]["document_id"] = "python_does_not_exist"
    errs = vq.validate_dataset(bad, corpus, combined=False)
    assert any("not found verbatim" in e for e in errs)
    assert any("unknown document_id" in e for e in errs)
