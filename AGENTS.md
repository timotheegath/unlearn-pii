# Agent Notes — unlearn-pii

This file gives an AI agent the context needed to contribute to this project
without re-deriving decisions already made.

## What this project is

A focused experiment evaluating machine unlearning techniques for PII in small
language models. The model used is GPT-2 (via HuggingFace). The goal is a clean
repo + short write-up, not a production system.

See `project.md` for the full scope, methodology, and decision log.

## Repository layout

unlearn-pii/
├── project.md        # Scope, methodology, status, decision log
├── agents.md         # This file
├── pyproject.toml    # uv-managed dependencies
├── data/             # Synthetic PII corpus (to be created)
├── notebooks/        # Exploration and evaluation notebooks (to be created)
└── src/              # Training and unlearning scripts (to be created)

## Environment

- Package manager: `uv` (`uv sync` to install, `uv run` to execute)
- Python: >=3.11
- Script/notebook project, not an installable package (`package = false`)
- No build backend required

## Key tool choices and why

| Tool | Role | Reason |
|---|---|---|
| `transformers` | GPT-2 model + tokenizer | Standard, GPT-2 loads in two lines |
| `datasets` | Corpus + forget/retain splits | Native split conventions for unlearning |
| `torch` | Training loop | Manual unlearning loops are plain PyTorch |
| `accelerate` | Device-agnostic training | Handles CPU/MPS/CUDA without code changes |
| `peft` | Optional LoRA | Makes edits reversible and cheap |
| `evaluate` + `scikit-learn` | MUSE metrics + MIA | Standard evaluation stack |
| `ruff` | Linting | Fast, zero-config |

## Implementation rules

- **Use libraries for**: model loading, tokenization, DataLoader, optimizer, LoRA
- **Write manually**: gradient ascent loop, negative finetuning, canary
  extraction, log-prob leakage checks, MUSE evaluation wrapper

Do not add dependencies without updating `pyproject.toml` and this file.

## Unlearning methods in scope

1. Gradient ascent on the forget set — negate the loss, standard SGD step
2. Negative finetuning — interleave ascent (forget) with descent (retain)

## Evaluation baseline

MUSE (Shi et al., 2024) + two extensions:
- Cross-version leakage: log-prob comparison before/after on forget vs holdout
- Mixed-query leakage: prompts blending retained context + forgotten PII

## Conventions

- All PII is synthetically generated and obviously fake
- Canary strings are unique and easy to grep
- Checkpoints saved before and after each unlearning run
- Seed all experiments; report across ≥2 random seeds
- Decision log lives in `project.md` — add an entry for every non-trivial choice

## What does not exist yet

- `data/` — synthetic PII corpus
- `src/train.py` — baseline fine-tuning script
- `src/unlearn.py` — gradient ascent + negative finetuning loops
- `src/evaluate.py` — MUSE metrics + leakage checks
- `notebooks/explore.ipynb` — exploration and plotting
