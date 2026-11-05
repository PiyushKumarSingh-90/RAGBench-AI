"""Phase 3 baseline tests. Use HashingEmbedder so no model download is needed."""

import re

import numpy as np
import pytest

from ragbench.chunking import chunk_corpus, fixed_size_chunks
from ragbench.corpus import Document, load_documents
from ragbench.embeddings import HashingEmbedder
from ragbench.generation import (
    ABSTAIN_TEXT,
    ExtractiveGenerator,
    GeneratedAnswer,
    build_prompt,
    parse_answer,
)
from ragbench.pipeline import BaselineConfig, RAGPipeline, build_index
from ragbench.retrieval import DenseRetriever
from ragbench.vectorstore import NumpyVectorStore


def make_doc(text, doc_id="d1"):
    return Document(doc_id, "python", "Title", "https://example.org/d1", text)


def test_fixed_chunks_size_overlap_and_coverage():
    doc = make_doc(" ".join(f"w{i}" for i in range(450)))
    chunks = fixed_size_chunks(doc, chunk_size=200, overlap=40)
    assert [len(c.text.split()) for c in chunks] == [200, 200, 130]
    assert chunks[1].text.split()[0] == "w160"
    assert chunks[-1].text.split()[-1] == "w449"
    assert len({c.chunk_id for c in chunks}) == 3


def test_chunk_heading_tracking_ignores_code_fences():
    text = "# Top\nintro words here\n## Section A\nalpha beta\n```\n# not a heading\n```\n"
    chunks = fixed_size_chunks(make_doc(text), chunk_size=3, overlap=0)
    headings = [c.section_heading for c in chunks]
    assert "Section A" in headings
    assert "not a heading" not in headings
    assert all("not a heading" not in c.headings_covered for c in chunks)
    spanning = fixed_size_chunks(make_doc(text), chunk_size=50, overlap=0)[0]
    assert spanning.section_heading == "Top"
    assert spanning.headings_covered == ("Top", "Section A")


def test_known_headings_survive_unbalanced_fence():
    text = "# Top\n```\ncode never closed\n## Real\nbody words\n# comment\n"
    doc = Document("d", "python", "Top", "u", text, headings=("Top", "Real"))
    chunk = fixed_size_chunks(doc, chunk_size=50, overlap=0)[0]
    assert chunk.headings_covered == ("Top", "Real")


def test_invalid_chunk_params():
    with pytest.raises(ValueError):
        fixed_size_chunks(make_doc("a b c"), chunk_size=10, overlap=10)


def test_real_corpus_chunks_keep_metadata():
    docs = load_documents()
    chunks = chunk_corpus(docs)
    by_id = {d.document_id: d for d in docs}
    assert {c.document_id for c in chunks} == set(by_id)
    for c in chunks:
        d = by_id[c.document_id]
        assert (c.source_url, c.document_title, c.source_name) == (
            d.source_url,
            d.title,
            d.source_name,
        )
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # Every stored heading that appears as a markdown heading line must be reachable.
    # (Docker page titles come from front-matter and have no heading line.)
    covered = {(c.document_id, h) for c in chunks for h in c.headings_covered}
    missing = [
        (d.document_id, h)
        for d in docs
        for h in d.headings
        if re.search(rf"^#{{1,6}}\s+{re.escape(h)}\s*$", d.content_markdown, re.M)
        and (d.document_id, h) not in covered
    ]
    assert missing == []


def _pipeline(texts, min_score=0.1):
    docs = [make_doc(t, f"d{i}") for i, t in enumerate(texts)]
    emb = HashingEmbedder()
    store = build_index(docs, emb, BaselineConfig(chunk_size=50, chunk_overlap=10, top_k=2))
    pipe = RAGPipeline(DenseRetriever(store, emb), ExtractiveGenerator(min_score=min_score), 2)
    pipe.store, pipe.embedder = store, emb
    return pipe


def test_pipeline_retrieves_and_cites_retrieved_chunk():
    pipe = _pipeline(
        [
            "Docker volumes persist data beyond the lifecycle of a container.",
            "Python lists support append and pop operations at the end.",
        ]
    )
    res = pipe.query("How do Docker volumes persist data?")
    assert res.retrieved[0].chunk.document_id == "d0"
    assert not res.answer.abstained
    retrieved_ids = {r.chunk.chunk_id for r in res.retrieved}
    assert res.citations and all(c["chunk_id"] in retrieved_ids for c in res.citations)
    assert {"embedding", "retrieval", "generation", "total"} <= set(res.timings_ms)


def test_pipeline_abstains_below_threshold():
    pipe = _pipeline(["Docker volumes persist data."], min_score=0.99)
    res = pipe.query("What is the airspeed of a swallow?")
    assert res.answer.abstained and res.answer.text == ABSTAIN_TEXT and res.citations == []


def test_pipeline_drops_fabricated_citations():
    class LyingGenerator:
        name = "liar"

        def generate(self, question, contexts, confidence=None):
            return GeneratedAnswer("x", [contexts[0].chunk.chunk_id, "fake::chunk"])

    pipe = _pipeline(["Docker volumes persist data."])
    pipe.generator = LyingGenerator()
    res = pipe.query("volumes")
    assert res.dropped_citations == ["fake::chunk"]
    assert [c["chunk_id"] for c in res.citations] == [res.retrieved[0].chunk.chunk_id]


def test_parse_answer_maps_markers_and_ignores_out_of_range():
    pipe = _pipeline(["alpha beta gamma delta", "epsilon zeta eta theta"])
    ctx = pipe.store.search(pipe.embedder.embed(["alpha"])[0], k=2)
    ans = parse_answer("Alpha is first [1][1][7].", ctx)
    assert ans.cited_chunk_ids == [ctx[0].chunk.chunk_id]
    assert parse_answer(ABSTAIN_TEXT, ctx).abstained
    prompt = build_prompt("q?", ctx)
    assert "[1]" in prompt and "[2]" in prompt and ABSTAIN_TEXT in prompt


def test_store_roundtrip(tmp_path):
    pipe = _pipeline(["Docker volumes persist data.", "Python lists."])
    pipe.store.save(tmp_path)
    loaded = NumpyVectorStore.load(tmp_path)
    assert loaded.meta == pipe.store.meta
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in pipe.store.chunks]
    assert np.array_equal(loaded.vectors, pipe.store.vectors)


def test_embedder_mismatch_rejected():
    pipe = _pipeline(["Docker volumes persist data."])
    with pytest.raises(ValueError):
        DenseRetriever(pipe.store, HashingEmbedder(dim=64))
