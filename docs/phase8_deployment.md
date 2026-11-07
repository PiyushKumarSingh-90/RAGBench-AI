# Phase 8 — Documentation, Results and Deployment

Status: complete, awaiting approval (2026-11-07).

## Documentation

`README.md` was rewritten as the public entry point. It covers the overview and
why RAGBench is an evaluation framework rather than a chatbot, the problem
statement, an architecture diagram (Mermaid), the four strategies, the corpus
and benchmark, the evaluation method, the results table, charts and findings,
dashboard screenshots, example questions and answers, setup, demo and
environment-variable instructions, testing and CI, limitations, and a phase index.

- **Preserved unchanged:** the Phase 1 sections (source selection, dataset summary, version notes, directory locations, regeneration, dataset limitations, and source attribution, licences and compliance).
- **Removed:** the outdated "Git Repository Status" section (it said 0 commits), and the per-phase run notes that the phase documents already hold.

Every number in the README comes from `results/phase5/metrics.json` or the Phase 4
results. The example answers are copied from `results/phase4_groq/hybrid/run.jsonl`;
the how-to answer is shortened with ellipses.

Screenshots (`docs/images/`) were taken on 2026-11-07 from the running dashboard with
the offline extractive generator: `dashboard_ask.jpg` (Ask tab) and
`dashboard_evaluation.jpg` (Evaluation results tab).

## Deployment

### Local (tested)

```bash
uvicorn api.main:app            # terminal 1
streamlit run dashboard/app.py  # terminal 2 -> http://localhost:8501
```

Tested in Phase 6, and again on 2026-11-07 for the screenshots.

### Docker (not tested)

| File | Purpose |
|---|---|
| `Dockerfile` | `python:3.11-slim`, CPU-only PyTorch, installs the package, builds both indexes at build time, and pre-downloads the reranker |
| `docker-compose.yml` | `api` on :8000 with a health check; `dashboard` on :8501, started once the API is healthy, reaching it at `http://api:8000` |
| `.dockerignore` | Keeps `.git`, `.venv`, `.env`, built indexes and raw docs out of the build context |
| `.env.example` | Template for the optional LLM keys; `.env` itself is git-ignored |

```bash
cp .env.example .env          # optional
docker compose up --build
```

**Docker is not installed on the development machine, so these files have never
been built or run.** The compose file parses as valid YAML. Before relying on it:
run `docker compose up --build`, open `http://localhost:8501`, and ask a question.
Expect a large image, mostly PyTorch and the models.

The image contains the processed corpus (`data/processed`) and the saved results.
It does not contain the raw documents. The licence and attribution notes in the
README apply to any public deployment. In particular, the AWS documentation is
CC-BY-SA-4.0: check the attribution requirements before you host it publicly.

## Verification (2026-11-07)

- Ruff lint and format: clean. `pytest`: 64 passed.
- The dashboard was run and checked in the browser: the Ask query returned an answer with sources and timings, and the Evaluation tab showed charts and tables.
- The README's numbers were checked against `results/phase5/metrics.json`.
- The Docker files were not built (Docker not available).
