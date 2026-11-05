# RAGBench — Phase 2 Benchmark Questions

**Phase:** 2 — Benchmark Question Creation
**Date:** November 2026
**Status:** All five per-source datasets reviewed and approved; combined benchmark created and validated (pending final Phase 2 approval)

---

## 1. Purpose

Phase 2 creates a fixed, manually verified set of 50 questions used to evaluate every retrieval strategy in later phases. Each answerable question carries exact evidence from the Phase 1 corpus (`data/processed/documents.jsonl`), so retrieval quality, answer correctness, faithfulness and citation support can be scored against ground truth. Unanswerable questions test whether a RAG system abstains instead of guessing.

No external sources were used. All expected answers and evidence come from the 65-document corpus.

## 2. Files

| File | Contents |
|---|---|
| `evaluation/datasets/python_questions.jsonl` | 10 Python questions |
| `evaluation/datasets/scikit_learn_questions.jsonl` | 10 scikit-learn questions |
| `evaluation/datasets/docker_questions.jsonl` | 10 Docker questions |
| `evaluation/datasets/fastapi_questions.jsonl` | 10 FastAPI questions |
| `evaluation/datasets/aws_questions.jsonl` | 10 AWS questions |
| `evaluation/datasets/benchmark_questions.jsonl` | All 50, the exact concatenation of the five files in the order above |
| `scripts/validate_questions.py` | Standard-library validator |
| `tests/test_questions.py` | Pytest wrapper for the validator plus a negative test |

## 3. Distribution

Each source has exactly 2 questions of each category (10 per source, 50 total).

| Category | Code | Per source | Total |
|---|---|---:|---:|
| Direct factual | `DIR` | 2 | 10 |
| Conceptual | `CON` | 2 | 10 |
| How-to | `HOW` | 2 | 10 |
| Multi-section | `MULTI` | 2 | 10 |
| Unanswerable | `UNANS` | 2 | 10 |

Question IDs follow `<SOURCE>-<TYPE>-<NN>` with source prefixes `PY`, `SK`, `DK`, `FA`, `AWS`.

Evidence summary: 82 evidence entries across 33 distinct documents. 4 multi-section questions cite two different documents (SK-MULTI-02, DK-MULTI-02, FA-MULTI-01, AWS-MULTI-02); the other 6 cite at least two distinct sections of one document.

## 4. Record schema

```json
{
  "question_id": "PY-DIR-01",
  "source_group": "python",
  "question_type": "direct_factual | conceptual | how_to | multi_section | unanswerable",
  "question": "...?",
  "expected_answer": "... or null",
  "answerable_from_corpus": true,
  "evidence": [
    {
      "document_id": "...",
      "source_name": "...",
      "document_title": "...",
      "section_heading": "...",
      "source_url": "...",
      "supporting_excerpt": "..."
    }
  ],
  "unanswerable_reason": null,
  "expected_behavior": "..."
}
```

Unanswerable records have `answerable_from_corpus: false`, `expected_answer: null`, `evidence: []`, a non-empty `unanswerable_reason`, and an `expected_behavior` that requires abstention.

## 5. Validation rules

`python scripts/validate_questions.py <file> [--combined]` checks:

- every line is valid JSON with all required fields;
- question IDs are unique and match the source/type pattern;
- exact counts: 10 records per source and 2 per category (50 and 5 sources with `--combined`);
- every evidence `document_id` exists, and `source_name`, `document_title` and `source_url` match the corpus record exactly;
- `section_heading` is one of the document's stored headings (stored text, including RST artifacts such as `!except`);
- `supporting_excerpt` occurs verbatim in the document's `content_markdown` (whitespace-normalized);
- multi-section questions cite at least 2 distinct (document, heading) pairs;
- unanswerable invariants (`false` / `null` / `[]` / reason present);
- no near-duplicate questions (token Jaccard similarity ≥ 0.8).

Automated checks cannot confirm that an excerpt *semantically* supports the answer. That was checked manually during the per-source reviews.

## 6. How unanswerable questions were verified

For each unanswerable question, the key terms were searched case-insensitively across **all 65 records** (not only the matching source). The `unanswerable_reason` field records the terms searched and any near-miss mentions that might tempt a system to guess, for example:

- SK-UNANS-01: `joblib` appears only in the context of caching and `pickle` only in the Python I/O tutorial.
- DK-UNANS-02: memory settings appear only in AWS Lambda/EC2 pages.
- FA-UNANS-02: `python-multipart` appears only in a dependency list.
- AWS-UNANS-01: the only nearby size limit is the 20 KB bucket policy limit.

## 7. Known limitations

- **Repeated heading names.** Some documents repeat headings (e.g. two `Usage` sections in `scikit_learn_modules_compose`; `Explanation`, `Note`, `Topics` across Docker/AWS pages). Evidence is identified by (document_id, section_heading), which is not always unique. Phase 5 retrieval scoring should match on document plus excerpt or heading, not on heading alone.
- **RST artifacts in Python headings and excerpts.** Some stored text contains markers such as `!except`. Evidence copies the stored text exactly. If `prepare_docs.py` is later changed to clean these, the corpus hash changes and the affected excerpts must be regenerated and re-reviewed.
- **FastAPI code examples are placeholders.** The FastAPI source Markdown includes code via `{* ../../docs_src/... *}` directives that were not expanded in Phase 1. Questions rely only on the prose, but no system can answer questions that need the code itself.
- **Uneven AWS coverage.** 4 of the 8 answerable AWS questions concern Lambda, and none concern EC2, whose collected pages are mostly feature lists and links.
- **Table-of-contents records.** Three scikit-learn records (`model_selection`, `supervised_learning`, `unsupervised_learning`) contain only toctree links and are not used as evidence.
- **Near-duplicate detection is lexical.** Paraphrased duplicates would not be caught automatically; the per-source reviews are the safeguard.

## 8. Change policy

After approval, the benchmark is frozen. Any change to a question, answer or evidence must be reviewed, and both the source file and `benchmark_questions.jsonl` must be updated together (a test enforces that the combined file equals the concatenation of the five source files).

SHA-256 at creation:

| File | SHA-256 |
|---|---|
| `benchmark_questions.jsonl` | `2926b3083e1afd574349797bf369e82e3a89c8af8eb064aacc5343f8dc682ca0` |
| `python_questions.jsonl` | `5c12f5c093cc647a50534800238a29e1beb3b89926ed6b32a884c52a6dd7944d` |
| `scikit_learn_questions.jsonl` | `e85205c9b811f08aa1caf5d9c4a557864f409c8e9c2b21d3059ecf50a047633e` |
| `docker_questions.jsonl` | `3e1bf0786277b0d150a90060a725e22b5cb41939b8ef1c9454a746b0b3952f2a` |
| `fastapi_questions.jsonl` | `cd45d840fd45452f27a4ec196bcf2413c3bad75180d6d7ec2136db510611c5d2` |
| `aws_questions.jsonl` | `683a9b8bbad5e9e1179d0fc796afc4977da67bbc0432b61e19945df3cc456e12` |
