# Speedrun RUNLOG

| Run | Hypothesis/Changes | Dev BPB Before | Dev BPB After | Conclusion |
|---|---|---|---|---|
| 1 | Baseline (Mediocre Trainer) | N/A | 2.3718 | Baseline established |
| 2 | Train BPE Tokenizer, Tie Weights, and Set n_embd=144, n_layer=5 | 2.3718 | N/A | Token compression improved 3.5x. Model params verified at 1.88M, fitting the budget constraints perfectly. |
| 3 | Use AdamW, Cosine Annealing, Gradient Accumulation (accum=2), and Grad Clip | N/A | 1.8134 | The architectural and training loop changes significantly improved training efficiency, resulting in a large drop in BPB while staying fully within the limits! |
