Retrieval over the sample corpus: 25 in-scope questions, embedding model `google/embeddinggemma-300m`.
A hit is a chunk from an evidence page.

| Mode | R@1 | R@3 | R@5 | R@8 | MRR |
|---|---|---|---|---|---|
| hybrid | 0.92 | 0.96 | 0.96 | 1.00 | 0.94 |
| vector | 0.92 | 0.96 | 0.96 | 1.00 | 0.94 |
| text | 0.76 | 0.80 | 0.80 | 0.80 | 0.77 |

Ranked below 3:

- hybrid: xl-uz-saturday-hours (rank 6)
- vector: xl-uz-saturday-hours (rank 6)
- text: xl-uz-saturday-hours (not in top 8), xl-ru-installments (not in top 8), xl-en-zirconia-crown (not in top 8), xl-en-root-canal (not in top 8), xl-ru-evening-shift (not in top 8)
