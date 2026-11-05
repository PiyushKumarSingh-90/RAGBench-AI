"""Build the Phase 3 baseline index (fixed-size chunks + dense embeddings).

python scripts/build_index.py                      # all-MiniLM-L6-v2
python scripts/build_index.py --embedder hashing   # offline, no torch
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ragbench.corpus import PROJECT_ROOT, load_documents
from ragbench.embeddings import DEFAULT_MODEL, HashingEmbedder, SentenceTransformerEmbedder
from ragbench.pipeline import BaselineConfig, build_index

DEFAULT_OUT = PROJECT_ROOT / "data" / "index" / "dense_fixed"


def make_embedder(kind: str, model: str):
    return HashingEmbedder() if kind == "hashing" else SentenceTransformerEmbedder(model)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", choices=["minilm", "hashing"], default="minilm")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-size", type=int, default=BaselineConfig.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=BaselineConfig.chunk_overlap)
    p.add_argument("--top-k", type=int, default=BaselineConfig.top_k)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()

    config = BaselineConfig(a.chunk_size, a.chunk_overlap, a.top_k)
    docs = load_documents()
    embedder = make_embedder(a.embedder, a.model)
    t0 = time.perf_counter()
    store = build_index(docs, embedder, config)
    store.meta["build_seconds"] = round(time.perf_counter() - t0, 2)
    store.save(a.out)
    print(f"Indexed {store.meta['num_chunks']} chunks from {len(docs)} documents -> {a.out}")
    for k, v in sorted(store.meta.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
