"""Phase 5: evaluate saved strategy runs with retrieval, answer and latency metrics.

    python scripts/evaluate.py                         # results/phase4_groq -> results/phase5
    python scripts/evaluate.py --run results/phase4    # any saved run directory

No API calls. Answer metrics use a local NLI cross-encoder and the MiniLM embedder
(downloaded from the Hugging Face Hub on first use). Outputs:

    <out>/<strategy>/eval.jsonl   per-question metrics
    <out>/metrics.json            aggregates per strategy
    <out>/report.md               tables
    <out>/charts/*.png            charts
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ragbench.corpus import PROJECT_ROOT
from ragbench.evaluation import DEFAULT_NLI_MODEL, NLIScorer, answer_metrics, retrieval_metrics

BENCHMARK = PROJECT_ROOT / "evaluation" / "datasets" / "benchmark_questions.jsonl"
STRATEGIES = ("dense_fixed", "dense_semantic", "hybrid", "rerank")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Categorical slots 1-4 of the dataviz reference palette, fixed order per strategy.
COLORS = {"dense_fixed": "#2a78d6", "dense_semantic": "#eb6834", "hybrid": "#1baf7a",
          "rerank": "#eda100"}  # fmt: skip


def load_chunk_text() -> dict[str, str]:
    text = {}
    for f in sorted((PROJECT_ROOT / "data" / "index").glob("*/chunks.jsonl")):
        for line in f.open(encoding="utf-8"):
            c = json.loads(line)
            text[c["chunk_id"]] = c["text"]
    return text


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1) if xs else None


def evaluate_strategy(rows, questions, chunk_text, nli, embedder) -> list[dict]:
    out = []
    answered = [
        r for r in rows if r["answerable_from_corpus"] and not r["abstained"] and r["answer"]
    ]
    sims = {}
    if answered:
        a = embedder.encode([r["answer"] for r in answered], normalize_embeddings=True)
        e = embedder.encode(
            [questions[r["question_id"]]["expected_answer"] for r in answered],
            normalize_embeddings=True,
        )
        sims = {r["question_id"]: float((x * y).sum()) for r, x, y in zip(answered, a, e)}
    for r in rows:
        q = questions[r["question_id"]]
        rec = {
            "question_id": r["question_id"],
            "question_type": r["question_type"],
            "source_group": r["source_group"],
            "answerable": r["answerable_from_corpus"],
            "abstained": r["abstained"],
            "timings_ms": r["timings_ms"],
        }
        if r["answerable_from_corpus"]:
            rec.update(retrieval_metrics(q, r["retrieved"], chunk_text))
            if r["question_id"] in sims:
                contexts = [chunk_text[x["chunk_id"]] for x in r["retrieved"]]
                rec.update(
                    answer_metrics(
                        r["answer"], q["expected_answer"], contexts, nli, sims[r["question_id"]]
                    )
                )
        else:
            rec["abstention_correct"] = r["abstained"]
            rec["top_confidence"] = r.get("confidence")
        out.append(rec)
    return out


def aggregate(evals: list[dict]) -> dict:
    ans = [e for e in evals if e["answerable"]]
    unans = [e for e in evals if not e["answerable"]]
    gen = [e for e in ans if "faithfulness" in e]
    agg = {
        "answerable": len(ans),
        "context_recall": mean(e["context_recall"] for e in ans),
        "evidence_fully_retrieved": mean(e["evidence_fully_retrieved"] for e in ans),
        "context_precision": mean(e["context_precision"] for e in ans),
        "map": mean(e["average_precision"] for e in ans),
        "hit_at_1": mean(float(e["hit_at_1"]) for e in ans),
        "hit_at_3": mean(float(e["hit_at_3"]) for e in ans),
        "hit_at_5": mean(float(e["hit_at_5"]) for e in ans),
        "answered": len(gen),
        "answerable_abstained": sum(e["abstained"] for e in ans),
        "faithfulness": mean(e["faithfulness"] for e in gen),
        "citation_support": mean(e["citation_support"] for e in gen),
        "citation_precision": mean(e["citation_precision"] for e in gen),
        "fact_coverage": mean(e["fact_coverage"] for e in gen),
        "token_f1": mean(e["token_f1"] for e in gen),
        "semantic_similarity": mean(e.get("semantic_similarity") for e in gen),
        "unanswerable": len(unans),
        "unanswerable_abstained": sum(e["abstention_correct"] for e in unans),
        "latency_ms": {},
        "by_type": {},
    }
    stages = sorted({s for e in evals for s in e["timings_ms"]})
    for s in stages:
        v = [e["timings_ms"][s] for e in evals if s in e["timings_ms"]]
        agg["latency_ms"][s] = {"median": pct(v, 0.5), "p95": pct(v, 0.95)}
    for t in sorted({e["question_type"] for e in ans}):
        sub = [e for e in ans if e["question_type"] == t]
        g = [e for e in sub if "faithfulness" in e]
        agg["by_type"][t] = {
            "n": len(sub),
            "context_recall": mean(e["context_recall"] for e in sub),
            "hit_at_5": mean(float(e["hit_at_5"]) for e in sub),
            "faithfulness": mean(e["faithfulness"] for e in g),
            "fact_coverage": mean(e["fact_coverage"] for e in g),
        }
    return agg


def fmt(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def report(metrics: dict, run_dir: Path, generator: str, nli_name: str) -> str:
    s = list(metrics)
    lines = [
        "# Phase 5 evaluation results",
        "",
        f"Run: `{run_dir.relative_to(PROJECT_ROOT)}` · generator: `{generator}` · "
        f"NLI judge: `{nli_name}` · similarity: `{EMBED_MODEL}`",
        "",
        "## Retrieval (40 answerable questions, top 5)",
        "",
        "| Strategy | Context recall | Evidence fully retrieved | Context precision | MAP | "
        "Hit@1 | Hit@3 | Hit@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in s:
        m = metrics[k]
        lines.append(
            f"| {k} | {fmt(m['context_recall'])} | {fmt(m['evidence_fully_retrieved'])} | "
            f"{fmt(m['context_precision'])} | {fmt(m['map'])} | {fmt(m['hit_at_1'])} | "
            f"{fmt(m['hit_at_3'])} | {fmt(m['hit_at_5'])} |"
        )
    lines += [
        "",
        "## Answers (answerable questions that were answered)",
        "",
        "| Strategy | Answered | Faithfulness | Citation support | Citation precision | "
        "Fact coverage | Token F1 | Semantic similarity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in s:
        m = metrics[k]
        lines.append(
            f"| {k} | {m['answered']}/{m['answerable']} | {fmt(m['faithfulness'])} | "
            f"{fmt(m['citation_support'])} | {fmt(m['citation_precision'])} | "
            f"{fmt(m['fact_coverage'])} | {fmt(m['token_f1'])} | {fmt(m['semantic_similarity'])} |"
        )
    lines += ["", "## Unanswerable questions", ""]
    lines += ["| Strategy | Correctly declined |", "|---|---:|"]
    for k in s:
        m = metrics[k]
        lines.append(f"| {k} | {m['unanswerable_abstained']}/{m['unanswerable']} |")
    stages = [
        st
        for st in (
            "embedding", "bm25", "fusion", "rerank", "retrieval_total", "generation", "total"
        )
        if any(st in metrics[k]["latency_ms"] for k in s)
    ]
    lines += [
        "",
        "## Latency (ms, median / p95, all 50 questions)",
        "",
        "| Strategy | " + " | ".join(stages) + " |",
        "|---|" + "---:|" * len(stages),
    ]
    for k in s:
        lat = metrics[k]["latency_ms"]
        cells = [
            f"{lat[st]['median']} / {lat[st]['p95']}" if st in lat else "–" for st in stages
        ]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    types = sorted({t for k in s for t in metrics[k]["by_type"]})
    lines += [
        "",
        "## By question type (context recall / fact coverage)",
        "",
        "| Strategy | " + " | ".join(types) + " |",
        "|---|" + "---:|" * len(types),
    ]
    for k in s:
        bt = metrics[k]["by_type"]
        lines.append(
            f"| {k} | "
            + " | ".join(
                f"{fmt(bt[t]['context_recall'], 2)} / {fmt(bt[t]['fact_coverage'], 2)}"
                for t in types
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def charts(metrics: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    strategies = list(metrics)
    ink, muted = "#1f1f1e", "#6b6a63"

    def style(ax):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c0")
        ax.tick_params(colors=muted, length=0)
        ax.xaxis.grid(True, color="#e6e5df", linewidth=0.8)
        ax.set_axisbelow(True)

    groups = [
        ("Retrieval quality", ["context_recall", "context_precision", "map", "hit_at_1"]),
        ("Answer quality", ["faithfulness", "citation_support", "fact_coverage", "token_f1"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, (title, keys) in zip(axes, groups):
        h = 0.18
        for i, st in enumerate(strategies):
            ys = [j + (i - 1.5) * (h + 0.02) for j in range(len(keys))]
            vals = [metrics[st][k] or 0 for k in keys]
            ax.barh(ys, vals, height=h, color=COLORS[st], label=st)
            for y, v in zip(ys, vals):
                ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=7, color=muted)
        ax.set_yticks(range(len(keys)), [k.replace("_", " ") for k in keys], color=ink)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.1)
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        style(ax)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, -0.22), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Retrieval latency is a separate chart: generation is ~100x larger (one axis per scale).
    fig, ax = plt.subplots(figsize=(7, 2.8))
    stages = [("embedding", "embedding"), ("bm25", "BM25"), ("fusion", "fusion"),
              ("rerank", "rerank")]  # fmt: skip
    stage_colors = ["#2a78d6", "#1baf7a", "#eb6834", "#eda100"]
    for i, st in enumerate(strategies):
        left = 0.0
        lat = metrics[st]["latency_ms"]
        for (key, label), c in zip(stages, stage_colors):
            if key in lat:
                v = lat[key]["median"]
                ax.barh(i, v, left=left, height=0.55, color=c, edgecolor="white", linewidth=2,
                        label=label if i == len(strategies) - 1 or key == "rerank" else None)
                left += v
        ax.text(left + 3, i, f"{lat['retrieval_total']['median']:.0f} ms", va="center",
                fontsize=8, color=muted)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.45))
    ax.set_yticks(range(len(strategies)), strategies, color=ink)
    ax.invert_yaxis()
    ax.set_title("Median retrieval latency by stage (ms)", loc="left", color=ink, fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "retrieval_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, default=PROJECT_ROOT / "results" / "phase4_groq")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "phase5")
    p.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    a = p.parse_args()

    from sentence_transformers import SentenceTransformer

    questions = {}
    for line in BENCHMARK.open(encoding="utf-8"):
        q = json.loads(line)
        questions[q["question_id"]] = q
    chunk_text = load_chunk_text()
    nli = NLIScorer(a.nli_model)
    embedder = SentenceTransformer(EMBED_MODEL)

    metrics, generator = {}, None
    for st in STRATEGIES:
        f = a.run / st / "run.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.open(encoding="utf-8") if line.strip()]
        generator = json.loads((a.run / st / "summary.json").read_text())["generator"]
        evals = evaluate_strategy(rows, questions, chunk_text, nli, embedder)
        (a.out / st).mkdir(parents=True, exist_ok=True)
        with (a.out / st / "eval.jsonl").open("w", encoding="utf-8") as fh:
            for e in evals:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        metrics[st] = aggregate(evals)
        print(f"{st}: recall={metrics[st]['context_recall']} faith={metrics[st]['faithfulness']}")

    meta = {"run": str(a.run.relative_to(PROJECT_ROOT)), "generator": generator,
            "nli_model": a.nli_model, "similarity_model": EMBED_MODEL}  # fmt: skip
    (a.out / "metrics.json").write_text(
        json.dumps({"meta": meta, "strategies": metrics}, indent=2) + "\n", encoding="utf-8"
    )
    md = report(metrics, a.run, generator, a.nli_model)
    (a.out / "report.md").write_text(md, encoding="utf-8")
    charts(metrics, a.out / "charts")
    print(md)


if __name__ == "__main__":
    main()
