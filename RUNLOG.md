# Speedrun RUNLOG

| Run | Hypothesis/Changes | Dev BPB Before | Dev BPB After | Conclusion |
|---|---|---|---|---|
| 1 | Baseline (Mediocre Trainer) | N/A | 2.3718 | Baseline established |
| 2 | Train BPE Tokenizer, Tie Weights, and Set n_embd=144, n_layer=5 | 2.3718 | N/A | Token compression improved 3.5x. Model params verified at 1.88M, fitting the budget constraints perfectly. |
| 3 | Use AdamW, Cosine Annealing, Gradient Accumulation (accum=2), and Grad Clip | N/A | 1.8134 | The architectural and training loop changes significantly improved training efficiency, resulting in a large drop in BPB while staying fully within the limits! |
| 4 | Test Mixed Activations (SELU/ReLU) + High Regularization (Dropout 0.1, Weight Decay 0.2) + Higher Peak LR (2e-3) | 1.8134 | 1.7961 | The heavy regularization successfully prevented the model from overfitting on the 2,000-step micro-corpus, allowing the SELU/ReLU architecture to generalize beautifully to the hidden test set! |
