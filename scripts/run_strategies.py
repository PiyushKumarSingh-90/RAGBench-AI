"""Run the four RAGBench retrieval strategies over the 50-question benchmark.

    python scripts/build_index.py --chunking fixed
    python scripts/build_index.py --chunking semantic
    python scripts/run_strategies.py                  # all four
    python scripts/run_strategies.py --strategies hybrid rerank
    python scripts/run_strategies.py --generator claude   # needs Anthropic credentials
    python scripts/run_strategies.py --generator gemini   # needs GEMINI_API_KEY
    python scripts/run_strategies.py --generator groq     # needs GROQ_API_KEY

Outputs go to results/phase4/<strategy>/{run.jsonl,summary.json} plus
results/phase4/comparison.{json,md}. The metrics here are *preliminary*
retrieval checks (document/section hit rates, MRR) and latency; the full
Phase 5 metric suite is not computed here.
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
from ragbench.retrieval import (
    DEFAULT_RERANKER,
    BM25Index,
    CrossEncoderScorer,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
)
from ragbench.vectorstore import NumpyVectorStore

BENCHMARK = PROJECT_ROOT / "evaluation" / "datasets" / "benchmark_questions.jsonl"
INDEX_ROOT = PROJECT_ROOT / "data" / "index"
OUT_ROOT = PROJECT_ROOT / "results" / "phase4"
STRATEGIES = ("dense_fixed", "dense_semantic", "hybrid", "rerank")
STAGES = (
    "embedding",
    "retrieval",
    "bm25",
    "fusion",
    "rerank",
    "retrieval_total",
    "generation",
    "total",
)


def load_embedder(store: NumpyVectorStore):
    model = store.meta["embedding_model"]
    return HashingEmbedder() if model.startswith("hashing") else SentenceTransformerEmbedder(model)


def score_row(q: dict, res: dict) -> dict:
    gold_docs = {e["document_id"] for e in q["evidence"]}
    gold_secs = {(e["document_id"], e["section_heading"]) for e in q["evidence"]}
    docs = [r["document_id"] for r in res["retrieved"]]
    secs = [{(r["document_id"], h) for h in r["headings_covered"]} for r in res["retrieved"]]
    first = next((i + 1 for i, d in enumerate(docs) if d in gold_docs), None)
    return {
        "doc_hit_at_1": bool(gold_docs & set(docs[:1])),
        "doc_hit_at_k": bool(gold_docs & set(docs)),
        "doc_mrr": 1 / first if first else 0.0,
        "section_hit_at_k": any(gold_secs & s for s in secs),
        "gold_docs_found": len(gold_docs & set(docs)),
        "gold_docs_total": len(gold_docs),
    }


def summarise(name: str, retriever_name: str, rows: list[dict], top_k: int) -> dict:
    ans = [r for r in rows if r["answerable_from_corpus"]]
    unans = [r for r in rows if not r["answerable_from_corpus"]]

    def mean(key, subset=ans):
        return round(statistics.mean(float(r[key]) for r in subset), 3)

    types = sorted({r["question_type"] for r in ans})
    return {
        "strategy": name,
        "retriever": retriever_name,
        "top_k": top_k,
        "num_answerable": len(ans),
        "num_unanswerable": len(unans),
        "doc_hit_at_1": mean("doc_hit_at_1"),
        f"doc_hit_at_{top_k}": mean("doc_hit_at_k"),
        f"doc_mrr_at_{top_k}": mean("doc_mrr"),
        f"section_hit_at_{top_k}": mean("section_hit_at_k"),
        "multi_section_gold_doc_coverage": round(
            sum(r["gold_docs_found"] for r in ans if r["question_type"] == "multi_section")
            / max(
                1, sum(r["gold_docs_total"] for r in ans if r["question_type"] == "multi_section")
            ),
            3,
        ),
        "section_hit_by_type": {
            t: f"{sum(r['section_hit_at_k'] for r in ans if r['question_type'] == t)}"
            f"/{sum(1 for r in ans if r['question_type'] == t)}"
            for t in types
        },
        "answerable_abstained": sum(r["abstained"] for r in ans),
        "unanswerable_abstained": sum(r["abstained"] for r in unans),
        "dropped_citations_total": sum(len(r["dropped_citations"]) for r in rows),
        "refusals": sum(r["generator_meta"].get("stop_reason") == "refusal" for r in rows),
        "empty_answers": sum(bool(r["generator_meta"].get("empty_answer")) for r in rows),
        "input_tokens_total": sum(r["generator_meta"].get("input_tokens", 0) for r in rows),
        "output_tokens_total": sum(r["generator_meta"].get("output_tokens", 0) for r in rows),
        "latency_ms_median": {
            s: round(statistics.median(r["timings_ms"][s] for r in rows), 3)
            for s in STAGES
            if s in rows[0]["timings_ms"]
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    p.add_argument("--min-score", type=float, default=0.35)
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--reranker", default=DEFAULT_RERANKER)
    p.add_argument(
        "--generator", choices=["extractive", "claude", "gemini", "groq"], default="extractive"
    )
    p.add_argument(
        "--model",
        default=None,
        help="LLM model name (each generator has its own default)",
    )
    p.add_argument("--effort", default=None, help="Claude effort (default: medium)")
    p.add_argument("--force", action="store_true", help="redo completed strategies")
    p.add_argument("--out", type=Path, default=None, help="default: results/phase4[_claude]")
    a = p.parse_args()

    fixed_store = NumpyVectorStore.load(INDEX_ROOT / "dense_fixed")
    embedder = load_embedder(fixed_store)
    top_k = fixed_store.meta["top_k"]
    dense_fixed = DenseRetriever(fixed_store, embedder, name="dense_fixed")

    retrievers = {}
    for name in a.strategies:
        if name == "dense_fixed":
            retrievers[name] = dense_fixed
        elif name == "dense_semantic":
            sem_store = NumpyVectorStore.load(INDEX_ROOT / "dense_semantic")
            if sem_store.meta["top_k"] != top_k:
                raise ValueError("fixed and semantic indexes must share top_k")
            retrievers[name] = DenseRetriever(sem_store, embedder, name="dense_semantic")
        elif name in ("hybrid", "rerank") and "_hybrid" not in retrievers:
            retrievers["_hybrid"] = HybridRetriever(
                dense_fixed, BM25Index(fixed_store.chunks), a.candidates, a.rrf_k
            )
        if name == "hybrid":
            retrievers[name] = retrievers["_hybrid"]
        if name == "rerank":
            retrievers[name] = RerankRetriever(
                retrievers["_hybrid"], CrossEncoderScorer(a.reranker), a.candidates
            )

    questions = [json.loads(line) for line in BENCHMARK.open(encoding="utf-8") if line.strip()]
    if a.generator == "claude":
        from ragbench.llm import DEFAULT_EFFORT, DEFAULT_MODEL, ClaudeGenerator

        generator = ClaudeGenerator(a.model or DEFAULT_MODEL, a.effort or DEFAULT_EFFORT)
    elif a.generator == "gemini":
        from ragbench.llm import GEMINI_DEFAULT_MODEL, GeminiGenerator

        generator = GeminiGenerator(a.model or GEMINI_DEFAULT_MODEL)
    elif a.generator == "groq":
        from ragbench.llm import GROQ_DEFAULT_MODEL, GroqGenerator

        generator = GroqGenerator(a.model or GROQ_DEFAULT_MODEL)
    else:
        generator = ExtractiveGenerator(min_score=a.min_score)
    llm = a.generator != "extractive"
    a.out = a.out or (OUT_ROOT.with_name(f"phase4_{a.generator}") if llm else OUT_ROOT)
    stopped = None
    for name in a.strategies:
        out = a.out / name
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "summary.json"
        if not a.force and summary_path.exists():
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            if prev.get("generator") == generator.name:
                print(f"{name}: already complete for {generator.name}; skipping (--force to redo)")
                continue
        # Resume: rows are appended to run.partial.jsonl as each question finishes.
        partial = out / "run.partial.jsonl"
        rows = []
        if partial.exists() and not a.force:
            rows = [json.loads(line) for line in partial.open(encoding="utf-8") if line.strip()]
            if rows and rows[0].get("generator") != generator.name:
                rows = []
            if rows:
                print(f"{name}: resuming after {len(rows)} saved questions")
        done = {r["question_id"] for r in rows}
        pipe = RAGPipeline(retrievers[name], generator, top_k)
        pipe.retriever.retrieve("warm-up query", top_k)  # exclude lazy init from latency
        with partial.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for q in questions:
                if q["question_id"] in done:
                    continue
                try:
                    res = pipe.query(q["question"]).to_dict()
                except Exception as e:  # e.g. daily quota exhausted: keep what we have
                    stopped = f"{name} at {q['question_id']}: {type(e).__name__}: {e}"
                    break
                row = {
                    "question_id": q["question_id"],
                    "source_group": q["source_group"],
                    "question_type": q["question_type"],
                    "answerable_from_corpus": q["answerable_from_corpus"],
                    "generator": generator.name,
                    **score_row(q, res),
                    **res,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        if stopped:
            print(f"\nSTOPPED ({len(rows)}/{len(questions)} saved for {name}). {stopped[:300]}")
            print("Rerun the same command later to resume from here.")
            break
        order = {q["question_id"]: i for i, q in enumerate(questions)}
        rows.sort(key=lambda r: order[r["question_id"]])
        with (out / "run.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        partial.unlink()
        s = summarise(name, retrievers[name].name, rows, top_k)
        s["generator"] = generator.name
        summary_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: doc@1={s['doc_hit_at_1']} section@{top_k}={s[f'section_hit_at_{top_k}']}")

    # The comparison covers every completed strategy in the output folder for this
    # generator, so strategies can be run in separate sessions.
    summaries = []
    for name in STRATEGIES:
        path = a.out / name / "summary.json"
        if path.exists():
            sm = json.loads(path.read_text(encoding="utf-8"))
            if sm.get("generator") == generator.name:
                summaries.append(sm)
    if not summaries:
        raise SystemExit(1 if stopped else 0)

    meta = {
        "embedding_model": fixed_store.meta["embedding_model"],
        "fixed_index": {
            k: fixed_store.meta[k] for k in ("num_chunks", "chunk_size", "chunk_overlap")
        },
        "candidates": a.candidates,
        "rrf_k": a.rrf_k,
        "reranker": a.reranker,
        "top_k": top_k,
        "generator": generator.name,
        "abstention_min_score": None if llm else a.min_score,
    }
    (a.out / "comparison.json").write_text(
        json.dumps({"config": meta, "strategies": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    k = top_k
    lines = [
        f"| Strategy | Doc hit@1 | Doc hit@{k} | Doc MRR@{k} | Section hit@{k} | "
        f"Multi-section gold-doc coverage | Unans. abstained | Median retrieval ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['strategy']} | {s['doc_hit_at_1']:.3f} | {s[f'doc_hit_at_{k}']:.3f} | "
            f"{s[f'doc_mrr_at_{k}']:.3f} | {s[f'section_hit_at_{k}']:.3f} | "
            f"{s['multi_section_gold_doc_coverage']:.3f} | "
            f"{s['unanswerable_abstained']}/{s['num_unanswerable']} | "
            f"{s['latency_ms_median']['retrieval_total']:.1f} |"
        )
    if llm:
        lines += [
            "",
            "| Strategy | Answerable abstained | Unanswerable abstained | Refusals | "
            "Input tokens | Output tokens | Median generation ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in summaries:
            lines.append(
                f"| {s['strategy']} | {s['answerable_abstained']}/{s['num_answerable']} | "
                f"{s['unanswerable_abstained']}/{s['num_unanswerable']} | {s['refusals']} | "
                f"{s['input_tokens_total']} | {s['output_tokens_total']} | "
                f"{s['latency_ms_median']['generation']:.0f} |"
            )
    (a.out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
