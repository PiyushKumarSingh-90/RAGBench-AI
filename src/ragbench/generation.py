"""Answer generation.

`Generator` is the interface a hosted LLM provider will implement later.
`build_prompt` / `parse_answer` define the grounded-answer contract such a
provider must follow: answer only from the numbered context, cite chunks as
[n], or reply with ABSTAIN_TEXT.

`ExtractiveGenerator` is a deterministic offline stand-in that needs no API key.
It is NOT an LLM: it selects the sentences from the retrieved context that
overlap most with the question, and abstains when retrieval confidence is low.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ragbench.vectorstore import SearchResult

ABSTAIN_TEXT = "The available documentation does not contain the answer to this question."

PROMPT_TEMPLATE = """You answer questions about technical documentation.
Use ONLY the numbered context passages below. Do not use outside knowledge.
Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
If the passages do not contain the answer, reply exactly:
{abstain}

Context:
{context}

Question: {question}
Answer:"""

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it of on or "
    "should so that the this to what when which why with you your".split()
)


@dataclass
class GeneratedAnswer:
    text: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[SearchResult]) -> GeneratedAnswer: ...


def build_prompt(question: str, contexts: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(contexts, 1):
        c = r.chunk
        blocks.append(f"[{i}] {c.document_title} — {c.section_heading} ({c.source_url})\n{c.text}")
    return PROMPT_TEMPLATE.format(
        abstain=ABSTAIN_TEXT, context="\n\n".join(blocks), question=question
    )


def parse_answer(raw: str, contexts: list[SearchResult]) -> GeneratedAnswer:
    """Turn raw LLM output into a GeneratedAnswer, mapping [n] markers to chunk IDs.

    Markers outside 1..len(contexts) are ignored, so a model cannot cite a passage
    it was not given.
    """
    text = raw.strip()
    if ABSTAIN_TEXT.lower() in text.lower():
        return GeneratedAnswer(text=ABSTAIN_TEXT, abstained=True)
    ids: list[str] = []
    for n in re.findall(r"\[(\d+)\]", text):
        i = int(n)
        if 1 <= i <= len(contexts):
            cid = contexts[i - 1].chunk.chunk_id
            if cid not in ids:
                ids.append(cid)
    return GeneratedAnswer(text=text, cited_chunk_ids=ids)


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOPWORDS}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]


class ExtractiveGenerator:
    """Offline, deterministic baseline generator (no LLM)."""

    def __init__(self, min_score: float = 0.35, max_sentences: int = 2):
        self.name = f"extractive(min_score={min_score},max_sentences={max_sentences})"
        self.min_score = min_score
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[SearchResult]) -> GeneratedAnswer:
        if not contexts or contexts[0].score < self.min_score:
            return GeneratedAnswer(text=ABSTAIN_TEXT, abstained=True)
        q = _content_tokens(question)
        candidates = []
        for r in contexts:
            for pos, s in enumerate(_sentences(r.chunk.text)):
                overlap = len(q & _content_tokens(s))
                if overlap:
                    # Prefer overlap, then higher-ranked chunk, then earlier sentence.
                    candidates.append((-overlap, r.rank, pos, s, r.chunk.chunk_id))
        if not candidates:
            return GeneratedAnswer(text=ABSTAIN_TEXT, abstained=True)
        picked = sorted(candidates)[: self.max_sentences]
        ids: list[str] = []
        parts = []
        for *_, s, cid in picked:
            if cid not in ids:
                ids.append(cid)
            parts.append(f"{s} [{ids.index(cid) + 1}]")
        return GeneratedAnswer(text=" ".join(parts), cited_chunk_ids=ids)
