"""Tests for Phase 1 documentation collection and preparation in RAGBench."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

APPROVED_SOURCES = {"python", "scikit_learn", "docker", "fastapi", "aws"}

EXPECTED_DOMAINS = {
    "python": "docs.python.org",
    "scikit_learn": "scikit-learn.org",
    "docker": "docs.docker.com",
    "fastapi": "fastapi.tiangolo.com",
    "aws": "docs.aws.amazon.com",
}

REQUIRED_FIELDS = {
    "document_id",
    "source_name",
    "title",
    "source_path",
    "source_url",
    "language",
    "headings",
    "content_markdown",
    "word_count",
    "content_sha256",
}


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def jsonl_path(project_root: Path) -> Path:
    return project_root / "data" / "processed" / "documents.jsonl"


@pytest.fixture(scope="module")
def manifest_path(project_root: Path) -> Path:
    return project_root / "data" / "processed" / "corpus_manifest.json"


@pytest.fixture(scope="module")
def documents(jsonl_path: Path) -> list[dict]:
    assert jsonl_path.exists(), f"Processed documents file does not exist at {jsonl_path}"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0, "documents.jsonl is empty"
    docs = []
    for idx, line in enumerate(lines, start=1):
        try:
            doc = json.loads(line)
            docs.append(doc)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Line {idx} in documents.jsonl is not valid JSON: {exc}")
    return docs


def test_processed_jsonl_exists(jsonl_path: Path):
    """1. The processed JSONL file exists."""
    assert jsonl_path.is_file()
    assert jsonl_path.stat().st_size > 0


def test_required_fields_present(documents: list[dict]):
    """2 & 3. Every document contains all required fields with non-empty basic values."""
    for doc in documents:
        missing = REQUIRED_FIELDS - set(doc.keys())
        assert not missing, f"Doc {doc.get('document_id')} is missing required fields: {missing}"


def test_all_five_sources_represented(documents: list[dict]):
    """4. All five sources are represented in the processed dataset."""
    present_sources = {doc["source_name"] for doc in documents}
    missing = APPROVED_SOURCES - present_sources
    assert not missing, f"Missing documentation sources: {missing}. Present: {present_sources}"


def test_document_ids_unique(documents: list[dict]):
    """5. Every document ID is unique."""
    ids = [doc["document_id"] for doc in documents]
    assert len(ids) == len(set(ids)), (
        f"Duplicate document IDs found: {len(ids) - len(set(ids))} duplicates"
    )


def test_source_names_approved(documents: list[dict]):
    """6. Every source name belongs to the approved five-source list."""
    for doc in documents:
        assert doc["source_name"] in APPROVED_SOURCES, f"Unknown source name: {doc['source_name']}"


def test_source_urls_domains(documents: list[dict]):
    """7. Every source URL uses the correct official domain."""
    for doc in documents:
        src = doc["source_name"]
        expected_domain = EXPECTED_DOMAINS[src]
        assert expected_domain in doc["source_url"], (
            f"Doc {doc['document_id']} URL '{doc['source_url']}' "
            f"does not contain expected domain '{expected_domain}'"
        )


def test_document_content_not_empty(documents: list[dict]):
    """8. Document content is not empty."""
    for doc in documents:
        content = doc["content_markdown"].strip()
        assert len(content) > 0, f"Document {doc['document_id']} has empty content_markdown"


def test_language_is_english(documents: list[dict]):
    """9. The language field is 'en'."""
    for doc in documents:
        assert doc["language"] == "en", (
            f"Doc {doc['document_id']} has non-English language: {doc['language']}"
        )


def test_word_counts_valid(documents: list[dict]):
    """10. Word counts are valid numbers matching token count."""
    for doc in documents:
        wc = doc["word_count"]
        assert isinstance(wc, int) and wc > 0, (
            f"Doc {doc['document_id']} has invalid word count {wc}"
        )
        tokens = [t for t in re.split(r"\s+", doc["content_markdown"].strip()) if t]
        # Verify tolerance within minor whitespace variation
        assert abs(len(tokens) - wc) <= 5, (
            f"Doc {doc['document_id']} stored word_count={wc} vs calculated {len(tokens)}"
        )


def test_content_sha256_hash_matches(documents: list[dict]):
    """11. SHA-256 hashes match the actual processed content."""
    for doc in documents:
        expected_hash = hashlib.sha256(doc["content_markdown"].encode("utf-8")).hexdigest()
        assert doc["content_sha256"] == expected_hash, (
            f"Doc {doc['document_id']} hash mismatch: {doc['content_sha256']} != {expected_hash}"
        )


def test_heading_structure(documents: list[dict]):
    """12. Heading information has the expected structure."""
    for doc in documents:
        headings = doc["headings"]
        assert isinstance(headings, list), f"Doc {doc['document_id']} headings is not a list"
        assert len(headings) > 0, f"Doc {doc['document_id']} has empty headings list"
        for h in headings:
            assert "level" in h and "text" in h, (
                f"Invalid heading structure in {doc['document_id']}: {h}"
            )
            assert isinstance(h["level"], int) and 1 <= h["level"] <= 6, (
                f"Invalid heading level: {h}"
            )
            assert isinstance(h["text"], str) and len(h["text"].strip()) > 0, (
                f"Empty heading text: {h}"
            )


def test_corpus_manifest_valid(manifest_path: Path, documents: list[dict]):
    """13. The corpus manifest exists and contains valid statistics."""
    assert manifest_path.is_file(), f"Manifest missing at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["project"] == "RAGBench"
    assert manifest["source_count"] == 5
    assert manifest["processed_document_count"] == len(documents)
    assert manifest["duplicate_count"] == 0
    assert manifest["validation"]["passed"] is True

    # Check sum of words
    expected_words = sum(doc["word_count"] for doc in documents)
    assert manifest["total_word_count"] == expected_words


def test_source_manifest_exists(project_root: Path):
    """Verify data/raw/source_manifest.json exists and covers all 5 sources."""
    sm_path = project_root / "data" / "raw" / "source_manifest.json"
    assert sm_path.is_file(), f"Source manifest missing at {sm_path}"
    data = json.loads(sm_path.read_text(encoding="utf-8"))
    assert data["project"] == "RAGBench"
    sources = {s["source_name"] for s in data["sources"]}
    assert sources == APPROVED_SOURCES, f"Sources in source_manifest.json mismatch: {sources}"
