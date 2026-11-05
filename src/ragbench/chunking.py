"""Document chunking.

Phase 3 implements only fixed-size chunking (the baseline). Each chunk keeps the
metadata needed to cite it: document ID, title, URL, the section heading in
effect where the chunk starts, and every heading the chunk spans.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ragbench.corpus import Document

HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*```")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_name: str
    document_title: str
    source_url: str
    section_heading: str
    chunk_index: int
    text: str
    headings_covered: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["headings_covered"] = list(self.headings_covered)
        return d


def _words_with_headings(
    markdown: str, default_heading: str, known_headings: tuple[str, ...] = ()
) -> list[tuple[str, str]]:
    """Split markdown into words, tagging each with the heading in effect.

    If `known_headings` is given (the document's stored heading list), a '#' line
    counts as a heading only when its text is in that list. This is robust to
    the unbalanced code fences some Phase 1 RST conversions contain. Otherwise
    '#' lines inside fenced code blocks are ignored.
    """
    known = set(known_headings)
    out: list[tuple[str, str]] = []
    heading = default_heading
    in_fence = False
    for line in markdown.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        m = HEADING_RE.match(line)
        if m:
            text = m.group(1)
            if (known and text in known) or (not known and not in_fence):
                heading = text
        out.extend((w, heading) for w in line.split())
    return out


def fixed_size_chunks(doc: Document, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split a document into windows of `chunk_size` words overlapping by `overlap` words."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")
    words = _words_with_headings(doc.content_markdown, doc.title, doc.headings)
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}::fixed::{idx:04d}",
                document_id=doc.document_id,
                source_name=doc.source_name,
                document_title=doc.title,
                source_url=doc.source_url,
                section_heading=window[0][1],
                chunk_index=idx,
                text=" ".join(w for w, _ in window),
                headings_covered=tuple(dict.fromkeys(h for _, h in window)),
            )
        )
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_corpus(docs: list[Document], chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    return [c for d in docs for c in fixed_size_chunks(d, chunk_size, overlap)]
