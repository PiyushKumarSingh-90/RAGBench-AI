# RAGBench — Phase 4 Retrieval Strategies

**Phase:** 4 — Four retrieval strategies
**Date:** November 2026
**Status:** Implemented and run on the 50-question benchmark; awaiting review

---

## 1. Strategies

All four strategies share the corpus loader, the `NumpyVectorStore`, the MiniLM embedder, the generator interface, citation guarding, top-k = 5, and one run script. Only the retriever differs.

| # | Name | Chunks | First stage | Final ranking |
|---|---|---|---|---|
| 1 | `dense_fixed` | fixed: 200 words, 40 overlap (972 chunks) | MiniLM cosine | top 5 by cosine |
| 2 | `dense_semantic` | section + paragraph, at most 200 words (1,268 chunks) | MiniLM cosine | top 5 by cosine |
| 3 | `hybrid` | fixed (same as 1) | BM25 top 20 + dense top 20 | Reciprocal Rank Fusion (k = 60), top 5 |
| 4 | `rerank` | fixed (same as 1) | hybrid top 20 | `cross-encoder/ms-marco-MiniLM-L6-v2` scores, top 5 |

Design choices (agreed before implementation):

- **Semantic chunking is structure-aware.** It never crosses a heading, packs whole paragraphs up to 200 words, and splits only a single paragraph that is longer than 200 words. Headings are recognised from each document's stored heading list, as in Phase 3. Every corpus word lands in exactly one chunk (verified: 152,638 words). 55 chunks are under 20 words (short sections).
- **BM25** is implemented in-house (Okapi, k1 = 1.5, b = 0.75, lowercase alphanumeric tokens, no stemming or stop-word removal), so no new dependency was added.
- **Hybrid fusion** uses Reciprocal Rank Fusion: score = Σ 1/(60 + rank) over the BM25 and dense rankings. Ties break by first appearance. RRF uses only ranks, so the two score scales never need normalising.
- **Reranking** applies the cross-encoder to the 20 hybrid candidates. That makes Strategy 4 "Strategy 3 + reranker", so its effect is measured against the hybrid result.
- **Abstention signal.** Each retriever reports a `confidence`, which is always the top dense cosine similarity. The offline extractive generator abstains below 0.35 for every strategy, so its abstention does not depend on the strategy's own score scale. This means abstention results are essentially the same for all four strategies and are **not** a strategy comparison.

## 2. Code

| File | Change |
|---|---|
| `src/ragbench/chunking.py` | Added `semantic_chunks` / `chunk_corpus_semantic` |
| `src/ragbench/retrieval.py` (new) | `DenseRetriever`, `BM25Index`, `reciprocal_rank_fusion`, `HybridRetriever`, `CrossEncoderScorer`, `RerankRetriever` |
| `src/ragbench/pipeline.py` | `RAGPipeline(retriever, generator, top_k)`; `BaselineConfig.chunking`; per-stage timings |
| `src/ragbench/generation.py` | `generate(..., confidence=None)` |
| `scripts/build_index.py` | `--chunking fixed|semantic` → `data/index/dense_<chunking>` |
| `scripts/run_strategies.py` (new) | Runs the strategies and writes `results/phase4/` |
| `scripts/run_baseline.py` | Adapted to the new pipeline; reproduces the Phase 3 numbers exactly |
| `tests/test_strategies.py` (new) | 10 tests, using a fake scorer and the hashing embedder (no downloads) |

Reproduce:

```bash
python scripts/build_index.py --chunking fixed
python scripts/build_index.py --chunking semantic
python scripts/run_strategies.py
```

## 3. Results (preliminary retrieval checks, 40 answerable questions)

From `results/phase4/comparison.md`:

| Strategy | Doc hit@1 | Doc hit@5 | Doc MRR@5 | Section hit@5 | Multi-section gold-doc coverage | Unans. abstained | Median retrieval ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_fixed | 0.925 | 1.000 | 0.954 | 0.925 | 0.857 | 1/10 | 5.9 |
| dense_semantic | **0.975** | 1.000 | **0.981** | 0.875 | **1.000** | 0/10 | 5.9 |
| hybrid | 0.925 | 1.000 | 0.958 | **1.000** | 0.929 | 1/10 | 7.1 |
| rerank | 0.925 | 1.000 | 0.947 | 0.975 | 0.857 | 1/10 | 183.2 |

Definitions:

- **Doc hit@k:** any gold evidence document appears in the top k.
- **Section hit@k:** a gold (document, section heading) pair appears in the `headings_covered` of any top-k chunk.
- **Multi-section gold-doc coverage:** the share of gold documents found, over the 10 multi-section questions (14 gold documents in total).

Misses:

| Strategy | Doc@1 misses | Section@5 misses |
|---|---|---|
| dense_fixed | DK-HOW-02, FA-DIR-01, AWS-MULTI-02 | FA-HOW-01, AWS-DIR-01, AWS-HOW-02 |
| dense_semantic | SK-MULTI-01 | PY-CON-01, DK-DIR-02, FA-DIR-01, FA-HOW-01, AWS-HOW-02 |
| hybrid | DK-HOW-02, FA-DIR-01, AWS-MULTI-02 | — |
| rerank | DK-HOW-02, FA-DIR-01, AWS-MULTI-02 | AWS-MULTI-02 |

Median latency per stage (ms, development Mac, models already loaded):

| Strategy | Embedding | Dense search | BM25 | Fusion | Rerank | Retrieval total |
|---|---:|---:|---:|---:|---:|---:|
| dense_fixed | 5.8 | 0.07 | – | – | – | 5.9 |
| dense_semantic | 5.8 | 0.09 | – | – | – | 5.9 |
| hybrid | 5.8 | 0.08 | 1.2 | 0.02 | – | 7.1 |
| rerank | 6.4 | 0.10 | 1.2 | 0.02 | 175.1 | 183.2 |

## 4. Observations

These come with a strong caveat: **with 40 answerable questions, one question equals 0.025**, and every strategy already reaches doc hit@5 = 1.0. The benchmark is near its ceiling at document level, so most differences are one or two questions. None of them are statistically significant.

- **Semantic chunking** ranks the right document first most often (39/40) and finds every gold document for multi-section questions. Its section hit rate is the lowest because each chunk carries a single heading, while fixed chunks span several. Section hit@k therefore favours chunks that span sections. Phase 5's evidence-based scoring should match excerpts, not just headings.
- **Hybrid** is the only strategy that hits a gold section for all 40 questions, at a cost of about +1.2 ms for BM25. Its doc@1 misses match dense_fixed, so the gain comes from what fills ranks 2–5.
- **Reranking** did not improve on hybrid here, and lost one section hit (AWS-MULTI-02). It adds about 175 ms per query, roughly 25× the retrieval cost. The ms-marco cross-encoder is trained on web search passages, so technical documentation is a domain shift for it. A larger candidate pool or a different reranker might change this, but neither was tried, to avoid tuning on the benchmark.
- **Abstention** is unchanged (0–1 of 10) because it uses the same dense-score threshold for every strategy. Separating strategies on unanswerable questions needs an LLM generator.
- **Gold-set completeness.** Some "misses" retrieve plausible but unlisted documents. For example, AWS-MULTI-02 retrieves `aws_s3_access_control`, which also discusses bucket policies. Gold evidence lists *sufficient* sources, not *all* relevant ones, so precision-style metrics in Phase 5 will be pessimistic.

## 5. Limitations

- No LLM is connected, so answer quality, faithfulness and citation support are not measured. The extractive stub's answers are copied text, not generated answers.
- The metrics are preliminary smoke checks. Context recall and precision, correctness, faithfulness and citation support are Phase 5.
- There is one run per strategy. Retrieval is deterministic for a fixed index and model versions (stable sorting, no sampling). Latencies vary between runs and machines.
- The settings (chunk size, overlap, candidates = 20, RRF k = 60, top-k = 5, threshold 0.35) are fixed defaults and were not tuned on the benchmark.
- Versions: numpy 2.4.6, sentence-transformers 6.2.0, torch 2.15.0, transformers 5.22.1; models `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L6-v2` (downloaded from Hugging Face on first use, revision not pinned).
