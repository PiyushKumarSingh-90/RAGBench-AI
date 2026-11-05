"""Phase 4 strategy tests. No model downloads: HashingEmbedder and a fake scorer."""

import numpy as np
import pytest

from ragbench.chunking import chunk_corpus, chunk_corpus_semantic, semantic_chunks
from ragbench.corpus import Document, load_documents
from ragbench.embeddings import HashingEmbedder
from ragbench.generation import ExtractiveGenerator
from ragbench.pipeline import BaselineConfig, RAGPipeline, build_index
from ragbench.retrieval import (
    BM25Index,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    reciprocal_rank_fusion,
)
from ragbench.vectorstore import SearchResult

TEXTS = [
    "Docker volumes persist data beyond the lifecycle of a container.",
    "Python lists support append and pop operations at the end of the list.",
    "The SimpleImputer replaces missing values with the column mean.",
    "FastAPI HTTPException returns an error response such as 404 to the client.",
]


def doc(text, i=0, headings=()):
    return Document(f"d{i}", "src", f"T{i}", f"https://x/{i}", text, headings)


def fixed_setup():
    docs = [doc(t, i) for i, t in enumerate(TEXTS)]
    emb = HashingEmbedder()
    store = build_index(docs, emb, BaselineConfig(chunk_size=50, chunk_overlap=10, top_k=2))
    return store, emb


# --- semantic chunking -------------------------------------------------------


def test_semantic_chunks_never_cross_headings_and_pack_paragraphs():
    text = "# A\n\np1 one two\n\np2 three four\n\n## B\n\np3 five six\n"
    chunks = semantic_chunks(doc(text, headings=("A", "B")), max_words=100)
    assert [c.section_heading for c in chunks] == ["A", "B"]
    assert chunks[0].text == "# A p1 one two p2 three four"
    assert all(c.headings_covered == (c.section_heading,) for c in chunks)


def test_semantic_chunks_respect_max_words_and_split_long_paragraphs():
    long_para = " ".join(f"w{i}" for i in range(25))
    text = f"# A\n\nshort para here\n\n{long_para}\n\nafter"
    chunks = semantic_chunks(doc(text, headings=("A",)), max_words=10)
    assert all(len(c.text.split()) <= 10 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == text.split()


def test_semantic_corpus_preserves_every_word():
    docs = load_documents()
    chunks = chunk_corpus_semantic(docs)
    assert sum(len(c.text.split()) for c in chunks) == sum(
        len(d.content_markdown.split()) for d in docs
    )
    assert all(len(c.text.split()) <= 200 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_build_index_semantic_meta():
    docs = [doc("# A\n\nalpha beta\n\n## B\n\ngamma", headings=("A", "B"))]
    store = build_index(docs, HashingEmbedder(), BaselineConfig(chunking="semantic"))
    assert store.meta["strategy"] == "dense_semantic" and store.meta["chunk_overlap"] == 0
    with pytest.raises(ValueError):
        build_index(docs, HashingEmbedder(), BaselineConfig(chunking="nope"))


# --- BM25 / RRF / hybrid -----------------------------------------------------


def test_bm25_prefers_rare_matching_terms():
    store, _ = fixed_setup()
    res = BM25Index(store.chunks).search("SimpleImputer missing values", 3)
    assert res[0].chunk.document_id == "d2"
    assert BM25Index(store.chunks).search("zzzz unknown", 3) == []


def test_bm25_on_real_corpus_finds_exact_identifier():
    chunks = chunk_corpus(load_documents())
    top = BM25Index(chunks).search("StratifiedKFold", 3)
    assert top and "StratifiedKFold" in top[0].chunk.text


def test_reciprocal_rank_fusion_rewards_agreement():
    store, _ = fixed_setup()
    c = store.chunks
    a = [SearchResult(c[0], 0.9, 1), SearchResult(c[1], 0.8, 2)]
    b = [SearchResult(c[1], 5.0, 1), SearchResult(c[2], 4.0, 2)]
    fused = reciprocal_rank_fusion([a, b], k=3, rrf_k=60)
    assert [r.chunk.document_id for r in fused] == ["d1", "d0", "d2"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert [r.rank for r in fused] == [1, 2, 3]


def test_hybrid_retriever_and_mismatch_guard():
    store, emb = fixed_setup()
    dense = DenseRetriever(store, emb)
    hybrid = HybridRetriever(dense, BM25Index(store.chunks), candidates=4)
    out = hybrid.retrieve("How does SimpleImputer handle missing values?", 2)
    assert out.results[0].chunk.document_id == "d2"
    assert {"embedding", "retrieval", "bm25", "fusion"} <= set(out.timings_ms)
    assert out.confidence == pytest.approx(
        dense.retrieve("How does SimpleImputer handle missing values?", 1).confidence
    )
    with pytest.raises(ValueError):
        HybridRetriever(dense, BM25Index(store.chunks[:2]))


# --- reranking ---------------------------------------------------------------


class FakeScorer:
    """Scores a text by whether it mentions 'FastAPI' (stands in for a cross-encoder)."""

    name = "fake"

    def score(self, question, texts):
        return np.array([1.0 if "FastAPI" in t else 0.0 for t in texts])


def test_rerank_reorders_candidates():
    store, emb = fixed_setup()
    dense = DenseRetriever(store, emb)
    rr = RerankRetriever(dense, FakeScorer(), candidates=4)
    out = rr.retrieve("Docker volumes persist data", 2)
    assert out.results[0].chunk.document_id == "d3"  # promoted by the scorer
    assert [r.rank for r in out.results] == [1, 2]
    assert "rerank" in out.timings_ms


def test_all_strategies_run_through_pipeline():
    store, emb = fixed_setup()
    dense = DenseRetriever(store, emb)
    hybrid = HybridRetriever(dense, BM25Index(store.chunks), candidates=4)
    for retriever in (dense, hybrid, RerankRetriever(hybrid, FakeScorer(), 4)):
        res = RAGPipeline(retriever, ExtractiveGenerator(min_score=0.1), 2).query(
            "Docker volumes persist data"
        )
        ids = {r.chunk.chunk_id for r in res.retrieved}
        assert len(res.retrieved) == 2
        assert all(c["chunk_id"] in ids for c in res.citations)
