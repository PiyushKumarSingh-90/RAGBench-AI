"""Text embedders. All return L2-normalised float32 matrices."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (default: all-MiniLM-L6-v2, 384 dims)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32, device=None):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _normalise(np.asarray(vecs))


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and offline use.

    Not a semantic model; it only exists so the pipeline can run without torch.
    """

    def __init__(self, dim: int = 256):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9_]+", t.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                m[i, h % self.dim] += 1.0
        return _normalise(m)
