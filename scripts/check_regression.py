"""Retrieval regression check for CI (offline: extractive generator, no API keys).

    python scripts/run_strategies.py --out /tmp/rb --force       # fresh retrieval run
    python scripts/check_regression.py /tmp/rb                    # compare to baseline
    python scripts/check_regression.py /tmp/rb --update           # accept as new baseline

For each strategy it compares retrieval quality with the committed baseline
(evaluation/regression_baseline.json). A metric that drops by more than the
tolerance fails the check; improvements are reported. Latency is reported but
never fails the check, because CI runners vary too much.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragbench.corpus import PROJECT_ROOT
from ragbench.evaluation import retrieval_metrics

BASELINE = PROJECT_ROOT / "evaluation" / "regression_baseline.json"
BENCHMARK = PROJECT_ROOT / "evaluation" / "datasets" / "benchmark_questions.jsonl"
QUALITY = (
    "doc_hit_at_1",
    "doc_hit_at_5",
    "doc_mrr_at_5",
    "section_hit_at_5",
    "context_recall",
    "evidence_hit_at_5",
)
DEFAULT_TOLERANCE = 0.03  # about one question out of 40


def chunk_texts() -> dict[str, str]:
    out = {}
    for f in sorted((PROJECT_ROOT / "data" / "index").glob("*/chunks.jsonl")):
        for line in f.open(encoding="utf-8"):
            c = json.loads(line)
            out[c["chunk_id"]] = c["text"]
    return out


def measure(run_dir: Path) -> dict:
    questions = {}
    for line in BENCHMARK.open(encoding="utf-8"):
        q = json.loads(line)
        questions[q["question_id"]] = q
    texts = chunk_texts()
    comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    result = {}
    for s in comparison["strategies"]:
        name = s["strategy"]
        rows = [json.loads(x) for x in (run_dir / name / "run.jsonl").open(encoding="utf-8")]
        ans = [r for r in rows if r["answerable_from_corpus"]]
        rm = [retrieval_metrics(questions[r["question_id"]], r["retrieved"], texts) for r in ans]
        result[name] = {
            "doc_hit_at_1": s["doc_hit_at_1"],
            "doc_hit_at_5": s["doc_hit_at_5"],
            "doc_mrr_at_5": s["doc_mrr_at_5"],
            "section_hit_at_5": s["section_hit_at_5"],
            "context_recall": round(sum(m["context_recall"] for m in rm) / len(rm), 4),
            "evidence_hit_at_5": round(sum(m["hit_at_5"] for m in rm) / len(rm), 4),
            "retrieval_ms_median": s["latency_ms_median"].get("retrieval_total"),
        }
    return result


def compare(current: dict, baseline: dict, tolerance: float) -> list[str]:
    failures = []
    print(f"{'strategy':<16}{'metric':<20}{'baseline':>10}{'current':>10}{'change':>10}")
    for name, base in baseline["strategies"].items():
        cur = current.get(name)
        if cur is None:
            failures.append(f"{name}: missing from current run")
            continue
        for k in QUALITY:
            delta = cur[k] - base[k]
            flag = "  FAIL" if delta < -tolerance else ("  better" if delta > tolerance else "")
            print(f"{name:<16}{k:<20}{base[k]:>10.3f}{cur[k]:>10.3f}{delta:>+10.3f}{flag}")
            if delta < -tolerance:
                failures.append(f"{name}.{k}: {base[k]:.3f} -> {cur[k]:.3f}")
        print(
            f"{name:<16}{'retrieval ms (info)':<20}{base['retrieval_ms_median']:>10}"
            f"{cur['retrieval_ms_median']:>10}"
        )
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--update", action="store_true", help="write the current run as baseline")
    a = p.parse_args()

    current = measure(a.run_dir)
    if a.update:
        tol = a.tolerance if a.tolerance is not None else DEFAULT_TOLERANCE
        a.baseline.write_text(
            json.dumps({"tolerance": tol, "strategies": current}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Baseline written to {a.baseline}")
        return
    baseline = json.loads(a.baseline.read_text(encoding="utf-8"))
    tol = a.tolerance if a.tolerance is not None else baseline.get("tolerance", DEFAULT_TOLERANCE)
    failures = compare(current, baseline, tol)
    if failures:
        print(f"\nREGRESSION (tolerance {tol}):\n  " + "\n  ".join(failures))
        sys.exit(1)
    print(f"\nNo regression (tolerance {tol}).")


if __name__ == "__main__":
    main()
