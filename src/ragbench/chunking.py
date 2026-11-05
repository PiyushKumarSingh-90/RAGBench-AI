"""Document chunking.

- fixed-size word windows (Strategy 1, the Phase 3 baseline);
- section + paragraph "semantic" chunks (Strategy 2).
 Each chunk keeps the
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


def _is_heading(line: str, known: set[str], in_fence: bool) -> str | None:
    m = HEADING_RE.match(line)
    if not m:
        return None
    text = m.group(1)
    if (known and text in known) or (not known and not in_fence):
        return text
    return None


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
        heading = _is_heading(line, known, in_fence) or heading
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


def _sections(doc: Document) -> list[tuple[str, list[str]]]:
    """Split a document into (heading, paragraphs). The heading line starts its section."""
    known = set(doc.headings)
    sections: list[tuple[str, list[str]]] = [(doc.title, [])]
    para: list[str] = []
    in_fence = False

    def flush():
        if para:
            sections[-1][1].append("\n".join(para))
            para.clear()

    for line in doc.content_markdown.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        heading = _is_heading(line, known, in_fence)
        if heading is not None:
            flush()
            sections.append((heading, []))
        if not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    return [(h, ps) for h, ps in sections if ps]


def semantic_chunks(doc: Document, max_words: int = 200) -> list[Chunk]:
    """Structure-aware chunks: never cross a heading; pack whole paragraphs up to
    `max_words`; split a single paragraph only if it alone exceeds `max_words`."""
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    pieces: list[tuple[str, str]] = []  # (heading, text)
    for heading, paragraphs in _sections(doc):
        buf: list[str] = []
        n = 0
        for p in paragraphs:
            words = p.split()
            if len(words) > max_words:
                if buf:
                    pieces.append((heading, " ".join(buf)))
                    buf, n = [], 0
                for i in range(0, len(words), max_words):
                    pieces.append((heading, " ".join(words[i : i + max_words])))
                continue
            if n + len(words) > max_words and buf:
                pieces.append((heading, " ".join(buf)))
                buf, n = [], 0
            buf.append(" ".join(words))
            n += len(words)
        if buf:
            pieces.append((heading, " ".join(buf)))
    return [
        Chunk(
            chunk_id=f"{doc.document_id}::semantic::{i:04d}",
            document_id=doc.document_id,
            source_name=doc.source_name,
            document_title=doc.title,
            source_url=doc.source_url,
            section_heading=h,
            chunk_index=i,
            text=t,
            headings_covered=(h,),
        )
        for i, (h, t) in enumerate(pieces)
    ]


def chunk_corpus_semantic(docs: list[Document], max_words: int = 200) -> list[Chunk]:
    return [c for d in docs for c in semantic_chunks(d, max_words)]
