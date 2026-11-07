"""Regression-check comparison logic (no indexes needed)."""

import importlib.util

from ragbench.corpus import PROJECT_ROOT

spec = importlib.util.spec_from_file_location(
    "check_regression", PROJECT_ROOT / "scripts" / "check_regression.py"
)
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def metrics(**over):
    m = {k: 0.9 for k in cr.QUALITY} | {"retrieval_ms_median": 5.0}
    return m | over


def test_drop_beyond_tolerance_fails_and_small_drop_passes():
    base = {"strategies": {"hybrid": metrics()}}
    assert cr.compare({"hybrid": metrics(context_recall=0.88)}, base, 0.03) == []
    fails = cr.compare({"hybrid": metrics(context_recall=0.85)}, base, 0.03)
    assert fails == ["hybrid.context_recall: 0.900 -> 0.850"]


def test_latency_never_fails_and_missing_strategy_fails():
    base = {"strategies": {"hybrid": metrics(), "rerank": metrics()}}
    fails = cr.compare({"hybrid": metrics(retrieval_ms_median=999.0)}, base, 0.03)
    assert fails == ["rerank: missing from current run"]


def test_committed_baseline_covers_all_strategies():
    import json

    base = json.loads(cr.BASELINE.read_text())
    assert set(base["strategies"]) == {"dense_fixed", "dense_semantic", "hybrid", "rerank"}
    assert all(set(cr.QUALITY) <= set(m) for m in base["strategies"].values())
