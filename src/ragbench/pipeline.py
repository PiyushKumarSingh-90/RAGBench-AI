"""RAG pipeline: retriever -> generator, with citation guarding and timings.

The retriever decides the strategy (see `ragbench.retrieval`); the generator
and citation handling are shared by all strategies.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ragbench.chunking import chunk_corpus, chunk_corpus_semantic
from ragbench.corpus import Document
from ragbench.embeddings import Embedder
from ragbench.generation import GeneratedAnswer, Generator
from ragbench.retrieval import DenseRetriever, Retriever
from ragbench.vectorstore import NumpyVectorStore, SearchResult

CHUNKINGS = ("fixed", "semantic")


@dataclass(frozen=True)
class BaselineConfig:
    """Index configuration. `chunk_overlap` applies to fixed chunking only."""

    chunk_size: int = 200
    chunk_overlap: int = 40
    top_k: int = 5
    chunking: str = "fixed"


@dataclass
class PipelineResult:
    question: str
    answer: GeneratedAnswer
    retrieved: list[SearchResult]
    citations: list[dict]
    confidence: float = 0.0
    dropped_citations: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer.text,
            "abstained": self.answer.abstained,
            "confidence": round(self.confidence, 6),
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
    if config.chunking == "fixed":
        chunks = chunk_corpus(docs, config.chunk_size, config.chunk_overlap)
    elif config.chunking == "semantic":
        chunks = chunk_corpus_semantic(docs, config.chunk_size)
    else:
        raise ValueError(f"chunking must be one of {CHUNKINGS}")
    vectors = embedder.embed([c.text for c in chunks])
    meta = {
        "strategy": f"dense_{config.chunking}",
        "embedding_model": embedder.name,
        "embedding_dim": int(vectors.shape[1]),
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        **asdict(config),
    }
    if config.chunking == "semantic":
        meta["chunk_overlap"] = 0
    return NumpyVectorStore(chunks, vectors, meta)


def config_from_meta(meta: dict) -> BaselineConfig:
    return BaselineConfig(
        chunk_size=meta["chunk_size"],
        chunk_overlap=meta["chunk_overlap"],
        top_k=meta["top_k"],
        chunking=meta.get("chunking", "fixed"),
    )


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    @classmethod
    def dense_from_index_dir(cls, directory: Path, embedder: Embedder, generator: Generator):
        store = NumpyVectorStore.load(directory)
        retriever = DenseRetriever(store, embedder, name=store.meta["strategy"])
        return cls(retriever, generator, store.meta["top_k"])

    def query(self, question: str) -> PipelineResult:
        t0 = time.perf_counter()
        out = self.retriever.retrieve(question, self.top_k)
        t1 = time.perf_counter()
        answer = self.generator.generate(question, out.results, confidence=out.confidence)
        t2 = time.perf_counter()

        # Never surface a citation to a chunk that was not retrieved.
        by_id = {r.chunk.chunk_id: r.chunk for r in out.results}
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
            **out.timings_ms,
            "retrieval_total": round((t1 - t0) * 1000, 3),
            "generation": round((t2 - t1) * 1000, 3),
            "total": round((t2 - t0) * 1000, 3),
        }
        return PipelineResult(
            question, answer, out.results, citations, out.confidence, dropped, timings
        )
