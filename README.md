# RAGBench — RAG Evaluation Framework

RAGBench is an evaluation framework designed to systematically compare and benchmark different Retrieval-Augmented Generation (RAG) retrieval strategies over technical documentation. By evaluating distinct retrieval and indexing pipelines against a curated ground-truth question dataset, RAGBench provides empirical insights into retrieval accuracy, synthesis quality, citation support, and latency trade-offs.

## Planned Retrieval Strategies

The framework will benchmark four distinct retrieval approaches:

1. **Dense Retrieval with Fixed-Size Chunking:** Simple token-based chunking paired with vector similarity search.
2. **Dense Retrieval with Semantic Chunking:** Adaptive chunking based on semantic boundaries paired with vector search.
3. **Hybrid Retrieval (BM25 + Dense):** Combination of sparse lexical matching (BM25) and dense semantic vector embeddings.
4. **Dense/Hybrid Retrieval with Reranking:** Initial candidate retrieval followed by a cross-encoder reranker.

## Evaluation Dimensions

- **Retrieval Quality:** Context recall, context precision, and hit rate at $k$.
- **Answer Correctness:** Semantic accuracy against reference answers.
- **Faithfulness & Citation Support:** Grounding of generated responses in retrieved context without hallucination.
- **Latency:** End-to-end response time and stage-specific retrieval latency.

## Development Status

> **Status: Phase 4 — Retrieval Strategies (implemented, awaiting review)**
>
> Phase 1 (corpus), Phase 2 (50-question benchmark) and Phase 3 (baseline pipeline) are complete. Phase 4 adds all four retrieval strategies behind one pipeline, plus LLM answer generators for Claude, Gemini and Groq (`--generator claude|gemini|groq`). None has completed a benchmark run yet; each needs an API key in the environment. The default generator is still the offline extractive stub. The full Phase 5 evaluation is not implemented.

---

## Phase 4 Retrieval Strategies

| # | Strategy | How it retrieves |
|---|---|---|
| 1 | `dense_fixed` | MiniLM embeddings over 200-word chunks with 40-word overlap |
| 2 | `dense_semantic` | MiniLM over section + paragraph chunks (no chunk crosses a heading) |
| 3 | `hybrid` | BM25 + dense, fused with Reciprocal Rank Fusion (k = 60) |
| 4 | `rerank` | Hybrid top 20, reranked by `cross-encoder/ms-marco-MiniLM-L6-v2` |

```bash
python scripts/build_index.py --chunking fixed
python scripts/build_index.py --chunking semantic
python scripts/run_strategies.py             # -> results/phase4/comparison.md

export ANTHROPIC_API_KEY=...                 # or: ant auth login
python scripts/run_strategies.py --generator claude   # claude-haiku-5-5 -> results/phase4_claude/

export GEMINI_API_KEY=...
python scripts/run_strategies.py --generator gemini   # gemini-flash-latest -> results/phase4_gemini/

export GROQ_API_KEY=...
python scripts/run_strategies.py --generator groq     # openai/gpt-oss-120b -> results/phase4_groq/
```

Preliminary results: every strategy finds a gold document in the top 5 for all 40 answerable questions. Semantic chunking ranks it first most often (0.975). Hybrid finds a gold section for 40/40. Reranking adds about 175 ms per query without improving on hybrid. See `docs/phase4_retrieval_strategies.md` for the full table, misses, latency and caveats.

---

## Phase 3 Baseline Pipeline

```bash
pip install -r requirements-dev.txt          # installs ragbench + numpy + sentence-transformers
python scripts/build_index.py                # 972 chunks -> data/index/dense_fixed (downloads MiniLM once)
python scripts/run_baseline.py               # runs 50 questions -> results/phase3_baseline/
```

Each answer includes citations (document, section, URL) restricted to retrieved chunks, plus per-stage latency. On the 40 answerable questions, the gold document is in the top 5 for all 40. Only 1 of 10 unanswerable questions is correctly abstained by the score-threshold stub. See `docs/phase3_baseline_pipeline.md` for configuration, results and limitations.

---

## Phase 2 Benchmark Questions

A manually verified set of **50 questions** (10 per source, with 2 each of: direct factual, conceptual, how-to, multi-section, unanswerable) in `evaluation/datasets/`. The five per-source files are `python_`, `scikit_learn_`, `docker_`, `fastapi_` and `aws_questions.jsonl`. `benchmark_questions.jsonl` is their exact concatenation.

- Every answerable question cites exact evidence from `data/processed/documents.jsonl`: document ID, title, URL, section heading, and a verbatim excerpt.
- Unanswerable questions were checked against all 65 documents and require the system to abstain.
- Validate with:

```bash
python scripts/validate_questions.py evaluation/datasets/benchmark_questions.jsonl --combined
```

See `docs/phase2_benchmark_questions.md` for the schema, validation rules, verification method and known limitations.

---

## Phase 1 Documentation Dataset

Phase 1 collected and structured official technical documentation across five essential technologies to build a robust, multi-domain knowledge corpus.

### Why These Sources Were Selected

1. **Python (`docs.python.org/3/`):** Core language fundamentals, data structures, OOP, standard library, and exception handling.
2. **Scikit-learn (`scikit-learn.org/stable/`):** Classical machine learning algorithms, preprocessing, pipelines, model selection, and evaluation metrics.
3. **Docker (`docs.docker.com/`):** Containerization, Dockerfile specifications, multi-stage builds, networking, storage, and Compose.
4. **FastAPI (`fastapi.tiangolo.com/`):** Modern API development, request validation, dependency injection, security, and asynchronous background tasks.
5. **AWS (`docs.aws.amazon.com/`):** Cloud infrastructure across four critical services: IAM, Amazon S3, Amazon EC2, and AWS Lambda.

### Dataset Summary

- **Collection Date:** November 2026 (UTC)
- **Total Processed Documents:** 65 pages
- **Total Corpus Word Count:** 152,638 words
- **Processed JSONL Size:** 1,244,227 bytes (~1.24 MB)
- **Chunking Status:** Unchunked (1 document = 1 complete official documentation page)

| Documentation Source | Repository Branch Snapshot | Content Revision | License Verification Revision | License | Pages Collected | Total Words |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **Python** | `cpython` (`main` branch) | `unknown`* | `0d3ab6f` | PSF-2.0 | 12 | 34,626 |
| **Scikit-learn** | `scikit-learn` (`main` branch) | `unknown`* | `d5a61d1` | BSD-3-Clause | 14 | 56,827 |
| **Docker** | `docker/docs` (`main` branch) | `unknown`** | `4937e25` | Apache-2.0 | 13 | 15,719 |
| **FastAPI** | `fastapi` (`master` branch) | `unknown`* | `80d475b` | MIT | 14 | 17,208 |
| **AWS** (IAM, S3, EC2, Lambda) | `docs.aws.amazon.com` (HTML) | `N/A` | `N/A` | CC-BY-SA-4.0 (docs) / MIT-0 (code) | 12 | 28,258 |
| **Total** | | | | | **65** | **152,638** |

*\*Note on Content Revision:* Files were downloaded via HTTP directly from raw branch endpoints without pinning an immutable commit SHA at fetch time; exact collection commit is recorded as `unknown` rather than assumed.  
*\*\*Note on Docker:* Candidate hash `a342371cfda92b772c57a9099859f7b19cecd315` was verified locally via Git against `docker/docs.git` and confirmed NOT to be a valid ref/object; original collection revision is recorded as `unknown`, while repository root `LICENSE` at commit `4937e25` was inspected and verified as Apache-2.0.

### Version Metadata Clarifications

- **Python:** The CPython repository default branch `main` tracks development towards Python 3.14. The tutorial documentation collected from `Doc/tutorial/` covers language fundamentals identical to the current stable Python 3.12/3.13 documentation published at `docs.python.org/3/`.
- **Scikit-learn:** The Scikit-learn repository default branch `main` tracks in-development releases (1.6/1.7 dev). The user guide chapters collected from `doc/` match the algorithms, preprocessing, and evaluation workflows documented in the current stable User Guide at `scikit-learn.org/stable/`.
- **FastAPI:** The default branch is `master`, which directly drives the deployed official documentation at `fastapi.tiangolo.com`. The collected pages represent the official English documentation for FastAPI 0.115+.
- **Docker:** The default branch is `main`, tracking the live documentation published at `docs.docker.com`.

### Directory Locations

- **Raw Documents:** `data/raw/<source_name>/` (contains original `.md`, `.rst`, `.html`, and license files).
- **Source Manifest:** `data/raw/source_manifest.json` (records provenance, repository URLs, commit SHAs, licenses, and collection metadata).
- **Processed Dataset:** `data/processed/documents.jsonl` (unified JSON Lines file containing structured documents).
- **Corpus Manifest:** `data/processed/corpus_manifest.json` (verified dataset statistics, word counts, and validation status).
- **Collection Notes:** `docs/phase1_collection_notes.md` (in-depth rationale, page mappings, and conversion details).

### How to Regenerate the Processed Dataset

The preparation script uses only Python standard-library modules and processes the raw files locally with zero network access:

```bash
python scripts/prepare_docs.py
```

Running this command is completely deterministic and reproduces the exact dataset and SHA-256 hashes.

### Running Verification Tests

Run the automated test suite verifying dataset integrity, required fields, unique IDs, SHA-256 hashes, heading structures, package imports, benchmark questions, and the baseline pipeline:

```bash
python -m pytest
```

Run code formatting and lint checks:

```bash
python -m ruff check .
python -m ruff format --check .
```

### Git Repository Status

- **Working Tree State:** All project files and directories are currently **untracked locally** (`??` in `git status`).
- **Commits:** **0 commits** have been made.
- **Remotes:** **No remote is configured** (no remote repository, no push performed).
- **Ignored Paths:** `.venv/`, `.env`, and cache directories (`.pytest_cache/`, `.ruff_cache/`) are excluded by `.gitignore`.

### Dataset Limitations

- **Scope:** Restricted to selected core conceptual and tutorial topics (10–15 pages per source) rather than complete exhaustive API reference libraries.
- **Language:** English documentation only.
- **Formats:** Converted from diverse source formats (Markdown, reStructuredText, HTML) into normalized GitHub-flavored Markdown. Navigation elements and boilerplate have been filtered out, but diagrams and images are represented by text references.

### Source Attribution, Licenses & Compliance

- **Python:** Python Software Foundation License Version 2 (PSF-2.0). Copyright © 2001–2026 Python Software Foundation.
- **Scikit-learn:** BSD 3-Clause License. Copyright © 2007–2026 scikit-learn developers.
- **Docker:** Apache License Version 2.0 (Apache-2.0). Copyright © 2016 Docker, Inc. Verified via Git clone at verified commit `4937e25`: Section 1 explicitly defines "Source" form to include "documentation source and configuration files", and Section 2 grants copyright license for the Work across both documentation source files and code examples.
- **FastAPI:** MIT License. Copyright © 2018–2026 Sebastián Ramírez.
- **AWS:** Documentation content licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0); sample code / code examples licensed under MIT No Attribution (MIT-0). Copyright © Amazon Web Services, Inc. or its affiliates.
  - *Technical Method vs. Legal Determination (AWS Site Terms):* Collection was conducted via a targeted, low-volume HTTP retrieval of 12 specific conceptual HTML pages across IAM, S3, EC2, and Lambda with an identified User-Agent (`RAGBench-Collector/1.0`) and inter-request delay intervals designed to minimize server impact. This technical practice is distinct from a legal determination that the collection complied with AWS Site Terms (e.g., Section 3 "Site Access and Content"). RAGBench makes no claim of explicit authorization, license waiver, or guaranteed legal compliance under AWS Site Terms. Users are responsible for evaluating their own compliance.
