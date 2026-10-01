Vector-only retrieval: 22 chunks (120 words max, 20 overlap), 23 in-scope questions.

| Model | Dim | R@1 | R@3 | R@5 | R@8 | MRR | Query ms |
|---|---|---|---|---|---|---|---|
| google/embeddinggemma-300m | 768 | 0.91 | 0.96 | 0.96 | 1.00 | 0.93 | 23 |
| minishlab/potion-multilingual-128M | 256 | 0.78 | 0.91 | 0.96 | 1.00 | 0.86 | 0 |
| Qwen/Qwen3-Embedding-0.6B-Q | 1024 | 0.74 | 0.91 | 1.00 | 1.00 | 0.83 | 773 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.57 | 0.74 | 0.96 | 1.00 | 0.71 | 5 |

Questions ranked below 3:

- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: hours-saturday (rank 4), complaint-response (rank 4), price-cleaning (rank 4), consultation-fee (rank 5), pensioner-discount (rank 6), xl-uz-saturday-hours (rank 4)
- minishlab/potion-multilingual-128M: pensioner-discount (rank 7), xl-uz-saturday-hours (rank 4)
- google/embeddinggemma-300m: xl-uz-saturday-hours (rank 6)
- Qwen/Qwen3-Embedding-0.6B-Q: parking (rank 4), complaint-response (rank 4)
