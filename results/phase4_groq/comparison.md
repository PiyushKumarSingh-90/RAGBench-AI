| Strategy | Doc hit@1 | Doc hit@5 | Doc MRR@5 | Section hit@5 | Multi-section gold-doc coverage | Unans. abstained | Median retrieval ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_fixed | 0.925 | 1.000 | 0.954 | 0.925 | 0.857 | 10/10 | 102.6 |
| dense_semantic | 0.975 | 1.000 | 0.981 | 0.875 | 1.000 | 10/10 | 101.7 |
| hybrid | 0.925 | 1.000 | 0.958 | 1.000 | 0.929 | 10/10 | 104.4 |
| rerank | 0.925 | 1.000 | 0.947 | 0.975 | 0.857 | 10/10 | 268.4 |

| Strategy | Answerable abstained | Unanswerable abstained | Refusals | Input tokens | Output tokens | Median generation ms |
|---|---:|---:|---:|---:|---:|---:|
| dense_fixed | 0/40 | 10/10 | 0 | 91546 | 13115 | 16450 |
| dense_semantic | 1/40 | 10/10 | 0 | 62308 | 12004 | 8363 |
| hybrid | 0/40 | 10/10 | 0 | 90121 | 12072 | 16130 |
| rerank | 0/40 | 10/10 | 0 | 91107 | 12028 | 16026 |
