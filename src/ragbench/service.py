"""Shared application service for the API: loads indexes, retrievers and generators once.

Models load lazily on first use and are cached for the life of the process.
LLM generators are offered only when their API key is in the environment.
"""

from __future__ import annotations

import json
import os
import threading
from functools import cached_property
from pathlib import Path

from ragbench.corpus import PROJECT_ROOT
from ragbench.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from ragbench.generation import ExtractiveGenerator
from ragbench.pipeline import RAGPipeline
from ragbench.retrieval import (
    DEFAULT_RERANKER,
    BM25Index,
    CrossEncoderScorer,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
)
from ragbench.vectorstore import NumpyVectorStore

STRATEGIES = {
    "dense_fixed": "Dense retrieval over fixed 200-word chunks (40-word overlap).",
    "dense_semantic": "Dense retrieval over section/paragraph chunks (max 200 words).",
    "hybrid": "BM25 + dense over fixed chunks, fused with Reciprocal Rank Fusion.",
    "rerank": "Hybrid top-20 re-scored by a cross-encoder.",
}
GENERATOR_KEYS = {
    "extractive": None,
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}


class RAGService:
    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = root
        self.index_root = root / "data" / "index"
        self.results_root = root / "results"
        self._lock = threading.Lock()
        self._retrievers: dict = {}
        self._generators: dict = {}

    # --- indexes -------------------------------------------------------------

    @cached_property
    def fixed_store(self) -> NumpyVectorStore:
        return NumpyVectorStore.load(self.index_root / "dense_fixed")

    @cached_property
    def semantic_store(self) -> NumpyVectorStore:
        return NumpyVectorStore.load(self.index_root / "dense_semantic")

    @cached_property
    def embedder(self):
        model = self.fixed_store.meta["embedding_model"]
        if model.startswith("hashing"):
            return HashingEmbedder()
        return SentenceTransformerEmbedder(model)

    @property
    def top_k(self) -> int:
        return self.fixed_store.meta["top_k"]

    @cached_property
    def _chunk_texts(self) -> dict[str, str]:
        stores = (self.fixed_store, self.semantic_store)
        return {c.chunk_id: c.text for s in stores for c in s.chunks}

    def chunk_text(self, chunk_id: str) -> str:
        return self._chunk_texts[chunk_id]

    # --- strategies and generators -------------------------------------------

    def retriever(self, strategy: str):
        if strategy not in STRATEGIES:
            raise KeyError(strategy)
        with self._lock:
            if strategy not in self._retrievers:
                dense = DenseRetriever(self.fixed_store, self.embedder, name="dense_fixed")
                if strategy == "dense_fixed":
                    r = dense
                elif strategy == "dense_semantic":
                    r = DenseRetriever(self.semantic_store, self.embedder, name="dense_semantic")
                else:
                    if "_hybrid" not in self._retrievers:
                        self._retrievers["_hybrid"] = HybridRetriever(
                            dense, BM25Index(self.fixed_store.chunks)
                        )
                    r = self._retrievers["_hybrid"]
                    if strategy == "rerank":
                        r = RerankRetriever(r, CrossEncoderScorer(DEFAULT_RERANKER))
                self._retrievers[strategy] = r
            return self._retrievers[strategy]

    def available_generators(self) -> dict[str, bool]:
        return {g: key is None or bool(os.environ.get(key)) for g, key in GENERATOR_KEYS.items()}

    def generator(self, name: str):
        if name not in GENERATOR_KEYS:
            raise KeyError(name)
        if not self.available_generators()[name]:
            raise PermissionError(f"{GENERATOR_KEYS[name]} is not set in the API's environment")
        with self._lock:
            if name not in self._generators:
                if name == "extractive":
                    g = ExtractiveGenerator()
                elif name == "groq":
                    from ragbench.llm import GroqGenerator

                    g = GroqGenerator()
                elif name == "gemini":
                    from ragbench.llm import GeminiGenerator

                    g = GeminiGenerator()
                else:
                    from ragbench.llm import ClaudeGenerator

                    g = ClaudeGenerator()
                self._generators[name] = g
            return self._generators[name]

    def query(self, question: str, strategy: str, generator: str, top_k: int | None = None):
        pipe = RAGPipeline(self.retriever(strategy), self.generator(generator), top_k or self.top_k)
        result = pipe.query(question).to_dict()
        result["strategy"] = strategy
        result["generator"] = self.generator(generator).name
        for r in result["retrieved"]:
            r["text"] = self.chunk_text(r["chunk_id"])
        return result

    # --- saved benchmark results ----------------------------------------------

    def benchmark_questions(self) -> list[dict]:
        path = self.root / "evaluation" / "datasets" / "benchmark_questions.jsonl"
        return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    def evaluation_metrics(self) -> dict | None:
        path = self.results_root / "phase5" / "metrics.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def benchmark_results(self, strategy: str) -> list[dict]:
        """Saved Groq answers for one strategy, merged with their Phase 5 metrics."""
        if strategy not in STRATEGIES:
            raise KeyError(strategy)
        run = self.results_root / "phase4_groq" / strategy / "run.jsonl"
        ev = self.results_root / "phase5" / strategy / "eval.jsonl"
        if not run.exists():
            return []
        evals = {}
        if ev.exists():
            for line in ev.open(encoding="utf-8"):
                e = json.loads(line)
                evals[e["question_id"]] = e
        rows = []
        for line in run.open(encoding="utf-8"):
            r = json.loads(line)
            r["evaluation"] = evals.get(r["question_id"])
            rows.append(r)
        return rows
