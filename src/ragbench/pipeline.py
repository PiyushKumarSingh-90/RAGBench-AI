"""Baseline RAG pipeline: fixed-size chunks -> dense embeddings -> top-k -> generator."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ragbench.chunking import chunk_corpus
from ragbench.corpus import Document
from ragbench.embeddings import Embedder
from ragbench.generation import GeneratedAnswer, Generator
from ragbench.vectorstore import NumpyVectorStore, SearchResult


@dataclass(frozen=True)
class BaselineConfig:
    chunk_size: int = 200
    chunk_overlap: int = 40
    top_k: int = 5


@dataclass
class PipelineResult:
    question: str
    answer: GeneratedAnswer
    retrieved: list[SearchResult]
    citations: list[dict]
    dropped_citations: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer.text,
            "abstained": self.answer.abstained,
            "citations": self.citations,
            "dropped_citations": self.dropped_citations,
            "retrieved": [
                {
                    "rank": r.rank,
                    "score": round(r.score, 6),
                    "chunk_id": r.chunk.chunk_id,
                    "document_id": r.chunk.document_id,
                    "section_heading": r.chunk.section_heading,
                    "headings_covered": list(r.chunk.headings_covered),
                }
                for r in self.retrieved
            ],
            "timings_ms": self.timings_ms,
        }


def build_index(
    docs: list[Document], embedder: Embedder, config: BaselineConfig
) -> NumpyVectorStore:
    chunks = chunk_corpus(docs, config.chunk_size, config.chunk_overlap)
    vectors = embedder.embed([c.text for c in chunks])
    meta = {
        "strategy": "dense_fixed",
        "embedding_model": embedder.name,
        "embedding_dim": int(vectors.shape[1]),
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        **asdict(config),
    }
    return NumpyVectorStore(chunks, vectors, meta)


class RAGPipeline:
    def __init__(
        self,
        store: NumpyVectorStore,
        embedder: Embedder,
        generator: Generator,
        config: BaselineConfig = BaselineConfig(),
    ):
        if store.meta.get("embedding_model") not in (None, embedder.name):
            raise ValueError(
                f"index built with {store.meta['embedding_model']!r}, "
                f"query embedder is {embedder.name!r}"
            )
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.config = config

    @classmethod
    def from_index_dir(cls, directory: Path, embedder, generator, config=None):
        store = NumpyVectorStore.load(directory)
        if config is None:
            config = BaselineConfig(
                chunk_size=store.meta["chunk_size"],
                chunk_overlap=store.meta["chunk_overlap"],
                top_k=store.meta["top_k"],
            )
        return cls(store, embedder, generator, config)

    def query(self, question: str) -> PipelineResult:
        t0 = time.perf_counter()
        qvec = self.embedder.embed([question])[0]
        t1 = time.perf_counter()
        retrieved = self.store.search(qvec, self.config.top_k)
        t2 = time.perf_counter()
        answer = self.generator.generate(question, retrieved)
        t3 = time.perf_counter()

        # Never surface a citation to a chunk that was not retrieved.
        by_id = {r.chunk.chunk_id: r.chunk for r in retrieved}
        valid = [cid for cid in answer.cited_chunk_ids if cid in by_id]
        dropped = [cid for cid in answer.cited_chunk_ids if cid not in by_id]
        answer.cited_chunk_ids = valid
        citations = [
            {
                "chunk_id": cid,
                "document_id": by_id[cid].document_id,
                "document_title": by_id[cid].document_title,
                "section_heading": by_id[cid].section_heading,
                "headings_covered": list(by_id[cid].headings_covered),
                "source_url": by_id[cid].source_url,
            }
            for cid in valid
        ]
        timings = {
            "embedding": (t1 - t0) * 1000,
            "retrieval": (t2 - t1) * 1000,
            "generation": (t3 - t2) * 1000,
            "total": (t3 - t0) * 1000,
        }
        return PipelineResult(
            question,
            answer,
            retrieved,
            citations,
            dropped,
            {k: round(v, 3) for k, v in timings.items()},
        )
