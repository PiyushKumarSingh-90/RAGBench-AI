"""Run the Phase 3 baseline over the 50-question benchmark.

Writes per-question outputs and a summary to results/phase3_baseline/.
The summary includes a *preliminary* document-level hit rate as a smoke check;
the full metric suite (context recall/precision, correctness, faithfulness,
citation support) is Phase 5.

    python scripts/run_baseline.py
    python scripts/run_baseline.py --index data/index/dense_fixed --min-score 0.35
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ragbench.corpus import PROJECT_ROOT
from ragbench.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from ragbench.generation import ExtractiveGenerator
from ragbench.pipeline import RAGPipeline
from ragbench.vectorstore import NumpyVectorStore

BENCHMARK = PROJECT_ROOT / "evaluation" / "datasets" / "benchmark_questions.jsonl"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "index" / "dense_fixed"
DEFAULT_OUT = PROJECT_ROOT / "results" / "phase3_baseline"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--min-score", type=float, default=0.35)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()

    meta = NumpyVectorStore.load(a.index).meta
    model = meta["embedding_model"]
    embedder = (
        HashingEmbedder() if model.startswith("hashing") else SentenceTransformerEmbedder(model)
    )
    generator = ExtractiveGenerator(min_score=a.min_score)
    pipe = RAGPipeline.dense_from_index_dir(a.index, embedder, generator)

    questions = [json.loads(line) for line in BENCHMARK.open(encoding="utf-8") if line.strip()]
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (a.out / "run.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            res = pipe.query(q["question"]).to_dict()
            gold = {e["document_id"] for e in q["evidence"]}
            ranked_docs = [r["document_id"] for r in res["retrieved"]]
            row = {
                "question_id": q["question_id"],
                "source_group": q["source_group"],
                "question_type": q["question_type"],
                "answerable_from_corpus": q["answerable_from_corpus"],
                "gold_document_ids": sorted(gold),
                "doc_hit_at_1": bool(gold & set(ranked_docs[:1])),
                "doc_hit_at_k": bool(gold & set(ranked_docs)),
                "top_score": res["retrieved"][0]["score"],
                **res,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ans = [r for r in rows if r["answerable_from_corpus"]]
    unans = [r for r in rows if not r["answerable_from_corpus"]]
    summary = {
        "index_meta": meta,
        "generator": generator.name,
        "num_questions": len(rows),
        "preliminary_doc_hit_at_1": round(sum(r["doc_hit_at_1"] for r in ans) / len(ans), 3),
        f"preliminary_doc_hit_at_{meta['top_k']}": round(
            sum(r["doc_hit_at_k"] for r in ans) / len(ans), 3
        ),
        "answerable_abstained": sum(r["abstained"] for r in ans),
        "unanswerable_abstained": sum(r["abstained"] for r in unans),
        "num_answerable": len(ans),
        "num_unanswerable": len(unans),
        "top_score_answerable_mean": round(statistics.mean(r["top_score"] for r in ans), 4),
        "top_score_unanswerable_mean": round(statistics.mean(r["top_score"] for r in unans), 4),
        "dropped_citations_total": sum(len(r["dropped_citations"]) for r in rows),
        "latency_ms_median": {
            k: round(statistics.median(r["timings_ms"][k] for r in rows), 3)
            for k in ("embedding", "retrieval", "generation", "total")
            if k in rows[0]["timings_ms"]
        },
        "by_question_type_doc_hit_at_k": {
            t: f"{sum(r['doc_hit_at_k'] for r in ans if r['question_type'] == t)}"
            f"/{sum(1 for r in ans if r['question_type'] == t)}"
            for t in sorted({r["question_type"] for r in ans})
        },
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
