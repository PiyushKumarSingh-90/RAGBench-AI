# Phase 7 — GitHub Actions and Regression Testing

Status: complete, awaiting approval (2026-11-07). The workflows have not run on
GitHub yet: the repository has no remote. Every step was run locally (below).

## Workflows

### `.github/workflows/ci.yml` — every push to `main`, every pull request

No API keys, no LLM calls, no cost.

| Job | Steps | Approx. time |
|---|---|---|
| `checks` | `ruff check`, `ruff format --check`, validate the five source question files and the combined benchmark (`scripts/validate_questions.py`), `pytest` | a few minutes (mostly installing CPU-only PyTorch) |
| `retrieval-regression` (after `checks`) | build both indexes (MiniLM, cached between runs), run all four strategies with the offline extractive generator, compare with the baseline, run the API tests against the built indexes, upload `comparison.md` and the regression report | about 5–10 minutes on a first run, less with the model cache |

### `.github/workflows/full-eval.yml` — manual only

Generates answers with Groq and grades them offline with `scripts/evaluate.py`;
uploads the run and the grades as an artifact. Needs the repository secret
`GROQ_API_KEY` (Settings → Secrets and variables → Actions). A full run needs
about 335k tokens, more than one day of Groq's free tier, so on the free tier it
is simpler to run locally and resume (Phase 4 §8). It is kept apart from the
pull-request checks so normal CI never spends tokens.

## Regression check

`scripts/check_regression.py <run_dir>` compares a fresh retrieval run with
`evaluation/regression_baseline.json`, per strategy:

| Metric | Source |
|---|---|
| `doc_hit_at_1`, `doc_hit_at_5`, `doc_mrr_at_5`, `section_hit_at_5` | Phase 4 comparison |
| `context_recall`, `evidence_hit_at_5` | Phase 5 evidence-excerpt metrics (`ragbench.evaluation.retrieval_metrics`) |

- A metric that drops by more than the tolerance (**0.03**, about one question out of 40) fails the job; larger improvements are marked `better`.
- Retrieval latency is printed for information only and never fails the job: shared CI runners vary too much.
- After an intended change (new chunking, new model), accept the new numbers with
  `python scripts/check_regression.py <run_dir> --update` and commit the baseline in the same pull request, so the change is visible in review.

The baseline was produced on 2026-11-07 from a fresh local run, and matches the
committed Phase 4/5 results (e.g. `hybrid` context recall 0.919, `rerank` 0.921).

## Local equivalent

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
python scripts/validate_questions.py --combined evaluation/datasets/benchmark_questions.jsonl
pytest -q
python scripts/run_strategies.py --out /tmp/rb --force
python scripts/check_regression.py /tmp/rb
```

## Verification (2026-11-07, local)

- Ruff lint and format: clean (69 files). Dataset validation: all five source files and the combined file PASSED.
- `pytest`: 64 passed. In a clean clone without indexes: 60 passed, 4 API tests skipped (the regression job runs them after building indexes).
- Fresh retrieval run + `check_regression.py`: no regression, all deltas 0.000.
- Workflow files parse as valid YAML. They have not been executed on GitHub.

## Changes in this phase

- `pyproject.toml`: `dev` extra (`pytest`, `ruff`).
- `scripts/evaluate.py`: accepts relative `--run`/`--out` paths (needed by the manual workflow).
- Two files reformatted with `ruff format` (no behaviour change).
