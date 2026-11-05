"""Retrievers for the four RAGBench strategies.

Every retriever returns a `RetrievalOutput`: ranked results, a `confidence`
value (always the top dense cosine similarity, so abstention behaves the same
way across strategies), and per-stage timings in milliseconds.

- DenseRetriever       Strategy 1 (fixed chunks) and Strategy 2 (semantic chunks)
- HybridRetriever      Strategy 3: BM25 + dense fused with Reciprocal Rank Fusion
- RerankRetriever      Strategy 4: cross-encoder reranking of hybrid candidates
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ragbench.chunking import Chunk
from ragbench.embeddings import Embedder
from ragbench.vectorstore import NumpyVectorStore, SearchResult

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _ms(t0: float, t1: float) -> float:
    return round((t1 - t0) * 1000, 3)


@dataclass
class RetrievalOutput:
    results: list[SearchResult]
    confidence: float
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int) -> RetrievalOutput: ...


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over chunk texts (lowercased alphanumeric tokens, no stemming)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) if len(chunks) else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self.tfs):
            for t, f in tf.items():
                self.postings.setdefault(t, []).append((i, f))

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.chunks), dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avgdl or 1))
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                s[i] += idf * f * (self.k1 + 1) / (f + norm[i])
        return s

    def search(self, query: str, k: int) -> list[SearchResult]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[: min(k, len(s))]
        return [
            SearchResult(self.chunks[i], float(s[i]), r + 1)
            for r, i in enumerate(order)
            if s[i] > 0
        ]


class DenseRetriever:
    def __init__(self, store: NumpyVectorStore, embedder: Embedder, name: str = "dense"):
        if store.meta.get("embedding_model") not in (None, embedder.name):
            raise ValueError(
                f"index built with {store.meta['embedding_model']!r}, "
                f"query embedder is {embedder.name!r}"
            )
        self.store, self.embedder, self.name = store, embedder, name

    def retrieve(self, question: str, k: int) -> RetrievalOutput:
        t0 = time.perf_counter()
        q = self.embedder.embed([question])[0]
        t1 = time.perf_counter()
        results = self.store.search(q, k)
        t2 = time.perf_counter()
        conf = results[0].score if results else 0.0
        return RetrievalOutput(results, conf, {"embedding": _ms(t0, t1), "retrieval": _ms(t1, t2)})


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]], k: int, rrf_k: int = 60
) -> list[SearchResult]:
    """Fuse rankings by sum of 1/(rrf_k + rank). Ties break by first appearance."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for r in ranking:
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + r.rank)
            chunks.setdefault(cid, r.chunk)
    order = sorted(scores, key=lambda c: -scores[c])  # sorted() is stable
    return [SearchResult(chunks[c], scores[c], i + 1) for i, c in enumerate(order[:k])]


class HybridRetriever:
    """BM25 and dense search over the same chunks, fused with RRF."""

    def __init__(
        self, dense: DenseRetriever, bm25: BM25Index, candidates: int = 20, rrf_k: int = 60
    ):
        dense_ids = [c.chunk_id for c in dense.store.chunks]
        if dense_ids != [c.chunk_id for c in bm25.chunks]:
            raise ValueError("BM25 and dense indexes must cover the same chunks")
        self.dense, self.bm25 = dense, bm25
        self.candidates, self.rrf_k = candidates, rrf_k
        self.name = f"hybrid(bm25+dense,rrf_k={rrf_k},candidates={candidates})"

    def retrieve(self, question: str, k: int) -> RetrievalOutput:
        d = self.dense.retrieve(question, self.candidates)
        t0 = time.perf_counter()
        sparse = self.bm25.search(question, self.candidates)
        t1 = time.perf_counter()
        fused = reciprocal_rank_fusion([d.results, sparse], k, self.rrf_k)
        t2 = time.perf_counter()
        timings = {**d.timings_ms, "bm25": _ms(t0, t1), "fusion": _ms(t1, t2)}
        return RetrievalOutput(fused, d.confidence, timings)


class CrossEncoderScorer:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device=None):
        from sentence_transformers import CrossEncoder  # heavy import, keep lazy

        self.name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def score(self, question: str, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.predict([(question, t) for t in texts]), dtype=np.float64)


class RerankRetriever:
    """Score first-stage candidates with a cross-encoder and keep the best k."""

    def __init__(self, base: Retriever, scorer, candidates: int = 20):
        self.base, self.scorer, self.candidates = base, scorer, candidates
        self.name = f"rerank({scorer.name}) over {base.name}, candidates={candidates}"

    def retrieve(self, question: str, k: int) -> RetrievalOutput:
        first = self.base.retrieve(question, self.candidates)
        t0 = time.perf_counter()
        scores = self.scorer.score(question, [r.chunk.text for r in first.results])
        order = np.argsort(-scores, kind="stable")[:k]
        t1 = time.perf_counter()
        results = [
            SearchResult(first.results[i].chunk, float(scores[i]), r + 1)
            for r, i in enumerate(order)
        ]
        return RetrievalOutput(
            results, first.confidence, {**first.timings_ms, "rerank": _ms(t0, t1)}
        )
