"""Recompute citations in saved LLM runs with the current `parse_answer` (offline).

Use after a citation-parsing fix, to update an existing run without new API calls:

    python scripts/reparse_citations.py results/phase4_groq

For each <run>/<strategy>/run.jsonl, citations and dropped_citations are rebuilt from
the stored answer text and the stored retrieved chunk IDs (in rank order); chunk
metadata comes from data/index/*/chunks.jsonl. Abstention and answers are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragbench.corpus import PROJECT_ROOT
from ragbench.generation import ABSTAIN_TEXT, parse_answer
from ragbench.vectorstore import NumpyVectorStore, SearchResult


def load_chunks() -> dict:
    chunks = {}
    for d in sorted((PROJECT_ROOT / "data" / "index").iterdir()):
        if (d / "chunks.jsonl").exists():
            chunks.update({c.chunk_id: c for c in NumpyVectorStore.load(d).chunks})
    return chunks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    a = p.parse_args()
    chunks = load_chunks()
    for run in sorted(a.run_dir.glob("*/run.jsonl")):
        rows = [json.loads(line) for line in run.open(encoding="utf-8") if line.strip()]
        before = sum(len(r["citations"]) for r in rows)
        for r in rows:
            if r["abstained"] or not r["answer"] or r["answer"] == ABSTAIN_TEXT:
                continue
            ctx = [
                SearchResult(chunks[x["chunk_id"]], x["score"], x["rank"]) for x in r["retrieved"]
            ]
            ids = parse_answer(r["answer"], ctx).cited_chunk_ids
            retrieved = {c.chunk.chunk_id: c.chunk for c in ctx}
            r["citations"] = [
                {
                    "chunk_id": cid,
                    "document_id": retrieved[cid].document_id,
                    "document_title": retrieved[cid].document_title,
                    "section_heading": retrieved[cid].section_heading,
                    "headings_covered": list(retrieved[cid].headings_covered),
                    "source_url": retrieved[cid].source_url,
                }
                for cid in ids
                if cid in retrieved
            ]
            r["dropped_citations"] = [cid for cid in ids if cid not in retrieved]
            r.setdefault("generator_meta", {})["citations_reparsed"] = True
        with run.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        after = sum(len(r["citations"]) for r in rows)
        uncited = sum(1 for r in rows if not r["abstained"] and r["answer"] and not r["citations"])
        print(f"{run.parent.name}: citations {before} -> {after}; uncited answers: {uncited}")


if __name__ == "__main__":
    main()
