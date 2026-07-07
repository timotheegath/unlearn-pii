from datasets import Dataset, DatasetDict
import random

def load_tiny_unlearning_dataset(
    forget_size=32,
    retain_size=96,
    val_size=32,
    seed=42,
):
    rng = random.Random(seed)

        # Load templates from files
    with open("data/retain_templates.txt", "r") as f:
        retain_templates = f.read().splitlines()
    
    with open("data/forget_templates.txt", "r") as f:
        forget_templates = f.read().splitlines()

    def build_rows(templates, size):
        rows = []
        for _ in range(size):
            t = rng.choice(templates)
            rows.append({"text": t.format(n=rng.randint(1, 999))})
        return rows

    return DatasetDict({
        "forget": Dataset.from_list(build_rows(forget_templates, forget_size)),
        "retain": Dataset.from_list(build_rows(retain_templates, retain_size)),
        "validation": Dataset.from_list(
            build_rows(forget_templates + retain_templates, val_size)
        ),
    })

def tokenize_for_causal_lm(dataset_dict, tokenizer, max_length=64):
    def tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset_dict.map(tok, batched=True, remove_columns=["text"])