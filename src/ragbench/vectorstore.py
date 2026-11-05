"""Exact cosine-similarity vector store backed by NumPy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragbench.chunking import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int


def _chunk_from_dict(d: dict) -> Chunk:
    d["headings_covered"] = tuple(d.get("headings_covered", ()))
    return Chunk(**d)


class NumpyVectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict | None = None):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.meta = meta or {}

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[SearchResult]:
        scores = self.vectors @ query_vector.astype(np.float32).reshape(-1)
        k = min(k, len(scores))
        # Stable sort on -score so ties break by index (deterministic).
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(order)
        ]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (directory / "index_meta.json").write_text(
            json.dumps(self.meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> NumpyVectorStore:
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy")
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [_chunk_from_dict(json.loads(line)) for line in f if line.strip()]
        meta = json.loads((directory / "index_meta.json").read_text(encoding="utf-8"))
        return cls(chunks, vectors, meta)
