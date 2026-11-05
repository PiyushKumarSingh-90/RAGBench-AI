"""Load the Phase 1 processed corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"


@dataclass(frozen=True)
class Document:
    document_id: str
    source_name: str
    title: str
    source_url: str
    content_markdown: str
    headings: tuple[str, ...] = ()


def load_documents(path: Path = DEFAULT_CORPUS_PATH) -> list[Document]:
    docs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            docs.append(
                Document(
                    document_id=r["document_id"],
                    source_name=r["source_name"],
                    title=r["title"],
                    source_url=r["source_url"],
                    content_markdown=r["content_markdown"],
                    headings=tuple(h["text"] for h in r.get("headings", [])),
                )
            )
    return docs
