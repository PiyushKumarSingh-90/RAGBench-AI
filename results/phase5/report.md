# Phase 5 evaluation results

Run: `results/phase4_groq` · generator: `groq(model=openai/gpt-oss-120b,temperature=0)` · NLI judge: `cross-encoder/nli-deberta-v3-base` · similarity: `sentence-transformers/all-MiniLM-L6-v2`

## Retrieval (40 answerable questions, top 5)

| Strategy | Context recall | Evidence fully retrieved | Context precision | MAP | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_fixed | 0.820 | 0.792 | 0.275 | 0.670 | 0.625 | 0.775 | 0.850 |
| dense_semantic | 0.775 | 0.757 | 0.220 | 0.593 | 0.475 | 0.750 | 0.825 |
| hybrid | 0.919 | 0.909 | 0.305 | 0.775 | 0.675 | 0.925 | 1.000 |
| rerank | 0.921 | 0.909 | 0.335 | 0.823 | 0.750 | 0.950 | 0.975 |

## Answers (answerable questions that were answered)

| Strategy | Answered | Faithfulness | Citation support | Citation precision | Fact coverage | Token F1 | Semantic similarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_fixed | 40/40 | 0.152 | 0.118 | 0.093 | 0.383 | 0.401 | 0.762 |
| dense_semantic | 39/40 | 0.274 | 0.219 | 0.185 | 0.438 | 0.432 | 0.791 |
| hybrid | 40/40 | 0.171 | 0.175 | 0.139 | 0.417 | 0.428 | 0.777 |
| rerank | 40/40 | 0.184 | 0.158 | 0.112 | 0.421 | 0.435 | 0.784 |

## Unanswerable questions

| Strategy | Correctly declined |
|---|---:|
| dense_fixed | 10/10 |
| dense_semantic | 10/10 |
| hybrid | 10/10 |
| rerank | 10/10 |

## Latency (ms, median / p95, all 50 questions)

| Strategy | embedding | bm25 | fusion | rerank | retrieval_total | generation | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_fixed | 97.0 / 307.0 | – | – | – | 97.3 / 309.6 | 16413.8 / 17550.4 | 16573.6 / 17712.3 |
| dense_semantic | 101.0 / 207.9 | – | – | – | 101.2 / 208.1 | 8335.5 / 17544.7 | 8406.0 / 17612.1 |
| hybrid | 96.1 / 307.4 | 2.5 / 10.3 | 0.0 / 0.1 | – | 104.1 / 310.0 | 16094.8 / 17803.5 | 16290.7 / 18041.7 |
| rerank | 65.4 / 111.7 | 1.7 / 4.1 | 0.0 / 0.1 | 198.4 / 209.3 | 268.0 / 313.9 | 15909.1 / 17649.7 | 16177.9 / 17921.5 |

## By question type (context recall / fact coverage)

| Strategy | conceptual | direct_factual | how_to | multi_section |
|---|---:|---:|---:|---:|
| dense_fixed | 0.90 / 0.42 | 0.99 / 0.30 | 0.71 / 0.42 | 0.67 / 0.40 |
| dense_semantic | 0.90 / 0.57 | 0.82 / 0.39 | 0.67 / 0.42 | 0.71 / 0.38 |
| hybrid | 0.96 / 0.42 | 1.00 / 0.42 | 0.95 / 0.42 | 0.77 / 0.42 |
| rerank | 0.95 / 0.48 | 1.00 / 0.37 | 0.95 / 0.50 | 0.78 / 0.33 |
