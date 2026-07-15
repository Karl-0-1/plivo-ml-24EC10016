# LLM Speedrun

This repository contains a highly optimized GPT architecture designed to maximize language modeling performance (BPB) under a strict 2,000-step and 2,000,000-parameter budget.

## Quick Start

Follow these instructions to clone the repository, install dependencies, and run the training/evaluation loop directly on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/Karl-0-1/plivo-ml-24EC10016.git
cd plivo-ml-24EC10016/llm_handout/starter
```

### 2. Install Dependencies
This project requires PyTorch and the HuggingFace `tokenizers` library for the custom 4,096-vocab BPE Tokenizer.
```bash
pip install torch tokenizers
```

### 3. Run the Training Loop
To train the model from scratch for the full 2,000 steps (this generates the `ckpt.pt` weight file):
```bash
python train.py --data ../data/train_corpus.txt --steps 2000
```
*(Note: This codebase is rigorously tested to run efficiently on standard CPUs to comply with submission requirements).*

### 4. Evaluate the Model
To test the finalized `ckpt.pt` against the evaluation dataset and output the final Bits Per Byte (BPB) score:
```bash
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

## Deliverables
All required assignment deliverables are included in this `starter` directory:
- `ckpt.pt`: The finalized 1.7961 BPB weight file.
- `RUNLOG.md`: The experimental run log.
- `NOTES.md`: A 10-sentence technical summary.
- `SUMMARY.html`: A polished overview of the architecture and workflow.
