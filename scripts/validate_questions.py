"""Validate RAGBench benchmark question datasets against the processed corpus.

Usage:
    python scripts/validate_questions.py evaluation/datasets/python_questions.jsonl
    python scripts/validate_questions.py evaluation/datasets/benchmark_questions.jsonl --combined

Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "processed" / "documents.jsonl"

SOURCE_PREFIX = {
    "python": "PY",
    "scikit_learn": "SK",
    "docker": "DK",
    "fastapi": "FA",
    "aws": "AWS",
}
TYPE_CODE = {
    "direct_factual": "DIR",
    "conceptual": "CON",
    "how_to": "HOW",
    "multi_section": "MULTI",
    "unanswerable": "UNANS",
}
REQUIRED_FIELDS = {
    "question_id",
    "source_group",
    "question_type",
    "question",
    "expected_answer",
    "answerable_from_corpus",
    "evidence",
    "unanswerable_reason",
    "expected_behavior",
}
EVIDENCE_FIELDS = {
    "document_id",
    "source_name",
    "document_title",
    "section_heading",
    "source_url",
    "supporting_excerpt",
}
PER_TYPE_PER_SOURCE = 2
NEAR_DUPLICATE_THRESHOLD = 0.8


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {r["document_id"]: r for r in map(json.loads, filter(str.strip, f))}


def validate_record(rec: dict, corpus: dict[str, dict]) -> list[str]:
    qid = rec.get("question_id", "<missing id>")
    errs: list[str] = []
    missing = REQUIRED_FIELDS - rec.keys()
    if missing:
        return [f"{qid}: missing fields {sorted(missing)}"]

    source, qtype = rec["source_group"], rec["question_type"]
    if source not in SOURCE_PREFIX:
        errs.append(f"{qid}: unknown source_group {source!r}")
    if qtype not in TYPE_CODE:
        errs.append(f"{qid}: unknown question_type {qtype!r}")
    if source in SOURCE_PREFIX and qtype in TYPE_CODE:
        pattern = rf"^{SOURCE_PREFIX[source]}-{TYPE_CODE[qtype]}-\d{{2}}$"
        if not re.match(pattern, qid):
            errs.append(f"{qid}: id does not match {pattern}")
    if not str(rec["question"]).strip().endswith("?"):
        errs.append(f"{qid}: question should end with '?'")
    if not str(rec["expected_behavior"]).strip():
        errs.append(f"{qid}: empty expected_behavior")

    if qtype == "unanswerable":
        if rec["answerable_from_corpus"] is not False:
            errs.append(f"{qid}: unanswerable must have answerable_from_corpus=false")
        if rec["expected_answer"] is not None:
            errs.append(f"{qid}: unanswerable must have expected_answer=null")
        if rec["evidence"] != []:
            errs.append(f"{qid}: unanswerable must have empty evidence")
        if not str(rec["unanswerable_reason"] or "").strip():
            errs.append(f"{qid}: unanswerable needs unanswerable_reason")
        return errs

    if rec["answerable_from_corpus"] is not True:
        errs.append(f"{qid}: answerable question must have answerable_from_corpus=true")
    if not str(rec["expected_answer"] or "").strip():
        errs.append(f"{qid}: empty expected_answer")
    if rec["unanswerable_reason"] is not None:
        errs.append(f"{qid}: answerable question must have unanswerable_reason=null")
    if not rec["evidence"]:
        errs.append(f"{qid}: answerable question has no evidence")

    for i, ev in enumerate(rec["evidence"]):
        tag = f"{qid} evidence[{i}]"
        if EVIDENCE_FIELDS - ev.keys():
            errs.append(f"{tag}: missing fields {sorted(EVIDENCE_FIELDS - ev.keys())}")
            continue
        doc = corpus.get(ev["document_id"])
        if doc is None:
            errs.append(f"{tag}: unknown document_id {ev['document_id']!r}")
            continue
        if ev["source_name"] != doc["source_name"]:
            errs.append(f"{tag}: source_name mismatch")
        if ev["source_name"] != source:
            errs.append(f"{tag}: evidence source differs from source_group")
        if ev["document_title"] != doc["title"]:
            errs.append(f"{tag}: title mismatch ({doc['title']!r})")
        if ev["source_url"] != doc["source_url"]:
            errs.append(f"{tag}: source_url mismatch ({doc['source_url']!r})")
        if ev["section_heading"] not in {h["text"] for h in doc["headings"]}:
            errs.append(f"{tag}: heading {ev['section_heading']!r} not in document")
        excerpt = normalize_ws(ev["supporting_excerpt"])
        if not excerpt or excerpt not in normalize_ws(doc["content_markdown"]):
            errs.append(f"{tag}: supporting_excerpt not found verbatim in document")

    if qtype == "multi_section":
        sections = {(e.get("document_id"), e.get("section_heading")) for e in rec["evidence"]}
        if len(sections) < 2:
            errs.append(f"{qid}: multi_section needs evidence from >=2 distinct sections")
    return errs


def validate_dataset(records: list[dict], corpus: dict[str, dict], combined: bool) -> list[str]:
    errs: list[str] = []
    for rec in records:
        errs.extend(validate_record(rec, corpus))

    ids = Counter(r.get("question_id") for r in records)
    errs += [f"duplicate question_id {q}" for q, n in ids.items() if n > 1]

    questions = [(r.get("question_id"), str(r.get("question", ""))) for r in records]
    for a in range(len(questions)):
        for b in range(a + 1, len(questions)):
            ta, tb = tokens(questions[a][1]), tokens(questions[b][1])
            sim = len(ta & tb) / max(1, len(ta | tb))
            if sim >= NEAR_DUPLICATE_THRESHOLD:
                errs.append(f"near-duplicate questions {questions[a][0]} / {questions[b][0]}")

    sources = Counter(r.get("source_group") for r in records)
    expected_sources = set(SOURCE_PREFIX) if combined else set(sources)
    if not combined and len(sources) != 1:
        errs.append(f"per-source file must contain one source_group, found {dict(sources)}")
    per_source = PER_TYPE_PER_SOURCE * len(TYPE_CODE)
    expected_total = per_source * len(expected_sources)
    if len(records) != expected_total:
        errs.append(f"expected {expected_total} records, found {len(records)}")
    for src in expected_sources:
        counts = Counter(r.get("question_type") for r in records if r.get("source_group") == src)
        for qtype in TYPE_CODE:
            if counts[qtype] != PER_TYPE_PER_SOURCE:
                errs.append(f"{src}: expected {PER_TYPE_PER_SOURCE} {qtype}, found {counts[qtype]}")
    return errs


def load_records(path: Path) -> tuple[list[dict], list[str]]:
    records, errs = [], []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errs.append(f"line {n}: invalid JSON ({exc})")
    return records, errs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--combined", action="store_true", help="validate the 50-question file")
    args = parser.parse_args(argv)

    records, errs = load_records(args.path)
    errs += validate_dataset(records, load_corpus(), args.combined)
    counts = Counter((r.get("source_group"), r.get("question_type")) for r in records)
    print(f"{args.path}: {len(records)} records")
    for (src, qtype), n in sorted(counts.items()):
        print(f"  {src:<13} {qtype:<15} {n}")
    if errs:
        print(f"FAILED with {len(errs)} error(s):")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
