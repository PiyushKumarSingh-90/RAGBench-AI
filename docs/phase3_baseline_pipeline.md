# RAGBench — Phase 3 Baseline RAG Pipeline

**Phase:** 3 — Baseline RAG pipeline
**Date:** November 2026
**Status:** Implemented and run on the 50-question benchmark; awaiting review

---

## 1. Decisions

| Decision | Choice | Reason |
|---|---|---|
| Generation | Pluggable `Generator` interface + offline `ExtractiveGenerator` stub | No API key yet. A hosted LLM can be added later without touching retrieval. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local) | Small (~90 MB), runs on 8 GB RAM, free and reproducible |
| Vector store | Exact cosine search in NumPy | 972 chunks fit easily in memory; deterministic; no extra service |

## 2. Architecture

```text
documents.jsonl ─► fixed-size chunker ─► MiniLM embeddings ─► NumPy store (data/index/dense_fixed)
                                                                    │
question ─► MiniLM embedding ─► top-k cosine search ─► Generator ─► answer + citations + timings
                                                                    │
                                     citations not in the retrieved set are dropped
```

| Module | Responsibility |
|---|---|
| `src/ragbench/corpus.py` | Load `documents.jsonl` (ID, title, URL, markdown, stored headings) |
| `src/ragbench/chunking.py` | Fixed-size word-window chunking with section metadata |
| `src/ragbench/embeddings.py` | `SentenceTransformerEmbedder`; `HashingEmbedder` (deterministic, tests only) |
| `src/ragbench/vectorstore.py` | `NumpyVectorStore`: search, save, load |
| `src/ragbench/generation.py` | `Generator` protocol, LLM prompt contract (`build_prompt` / `parse_answer`), `ExtractiveGenerator` |
| `src/ragbench/pipeline.py` | `build_index`, `RAGPipeline.query` |
| `scripts/build_index.py` | Build and save the index |
| `scripts/run_baseline.py` | Run all 50 questions and write `results/phase3_baseline/` |
| `tests/test_pipeline.py` | 11 tests; uses `HashingEmbedder`, so no model download needed |

## 3. Configuration (recorded in `index_meta.json` and `summary.json`)

| Setting | Value |
|---|---|
| Chunking | fixed-size, 200 words, 40-word overlap |
| Chunks | 972 from 65 documents |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, 384 dims, normalised |
| Similarity | cosine (dot product of normalised vectors), exact, ties broken by index |
| top-k | 5 |
| Generator | `extractive(min_score=0.35, max_sentences=2)` |
| Library versions | numpy 2.4.6, sentence-transformers 6.2.0, torch 2.15.0, transformers 5.22.1 |

### Chunk metadata

Each chunk records `chunk_id` (`<document_id>::fixed::<NNNN>`), `document_id`, `source_name`, `document_title`, `source_url`, `section_heading` (the heading where the chunk starts) and `headings_covered` (every heading the chunk spans).

A heading is recognised only when a `#` line matches one of the document's stored `headings`. This was needed because 8 Phase 1 documents (3 Python, 5 scikit-learn) contain unbalanced code fences left by the RST conversion. Plain fence-tracking would have swallowed real headings such as "Source Code Encoding". The corpus itself was not modified.

## 4. Grounding and citations

- `build_prompt` gives the LLM numbered passages with title, section and URL. It instructs the model to use only those passages, cite them as `[n]`, or reply with a fixed abstention sentence.
- `parse_answer` maps `[n]` markers to chunk IDs and ignores out-of-range numbers.
- `RAGPipeline.query` drops any cited chunk ID that was not retrieved (reported as `dropped_citations`). A generator therefore cannot surface a fabricated citation.
- `ExtractiveGenerator` is **not an LLM**. It returns the two retrieved sentences with the most word overlap with the question, each cited. It abstains when the top retrieval score is below `min_score`. Its answers are verbatim excerpts, not fluent answers. It exists so the pipeline runs end to end without an API key.

## 5. Baseline run (`results/phase3_baseline/`)

`python scripts/build_index.py && python scripts/run_baseline.py`

| Measure (preliminary) | Result |
|---|---|
| Gold document in top-1 (40 answerable) | 0.925 (37/40) |
| Gold document in top-5 | 1.000 (40/40) |
| Gold (document, section) in a top-5 chunk's `headings_covered` | 37/40 (misses: FA-HOW-01, AWS-DIR-01, AWS-HOW-02) |
| Answerable questions wrongly abstained | 0/40 |
| Unanswerable questions correctly abstained | **1/10** |
| Mean top-1 score: answerable / unanswerable | 0.662 / 0.501 |
| Fabricated citations dropped | 0 |
| Median latency (ms): embed / retrieve / generate / total | 5.6 / 0.06 / 0.31 / 6.0 |

The document hit rates above are smoke checks only. The planned metrics (context recall and precision, answer correctness, faithfulness, citation support) belong to Phase 5. Latencies were measured on the development Mac (Apple Silicon, 8 GB) and exclude model loading. Generation latency is for the extractive stub and says nothing about an LLM.

### Abstention finding

The retrieval similarity score is a weak abstention signal. Answerable top-1 scores range from 0.538 upward, while unanswerable ones reach 0.595. A diagnostic sweep shows the trade-off:

| `min_score` | Answerable wrongly abstained | Unanswerable correctly abstained |
|---|---:|---:|
| 0.35 (default) | 0/40 | 1/10 |
| 0.50 | 0/40 | 6/10 |
| 0.53 | 0/40 | 7/10 |
| 0.55 | 2/40 | 7/10 |

The default was **not** tuned on the benchmark: choosing a threshold from these 50 questions would overfit the test set. Reliable abstention is expected to come from the LLM reading the context, under the prompt contract above, rather than from a score cut-off.

## 6. Limitations

- No LLM is connected, so answer quality cannot yet be judged.
- The `section_heading` of a chunk is where it starts. Use `headings_covered` for section-level matching.
- Fixed word windows split sentences and code blocks mid-way. This is expected for the baseline; semantic chunking is Phase 4.
- `data/index/` is git-ignored and must be rebuilt with `scripts/build_index.py`. The first run downloads the MiniLM model from Hugging Face.
- Retrieval uses whole-document word counts. FastAPI chunks contain the unexpanded code-include placeholders noted in Phase 2.

## 7. Next steps (not started)

- Connect a hosted LLM via the `Generator` interface once a provider and key are chosen.
- Phase 4: semantic chunking, hybrid BM25 + dense, and cross-encoder reranking, all reusing this corpus loader, store, generator interface and run script.
