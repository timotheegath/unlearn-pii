from datasets import Dataset, DatasetDict
import random

def load_tiny_unlearning_dataset(
    forget_size=32,
    retain_size=96,
    val_size=32,
    seed=42,
):
    rng = random.Random(seed)

    forget_templates = [
        "User email is alice{n}@example.com.",
        "Home address: {n} King Street, London.",
        "Contact bob{n}@mail.com for billing.",
        "Shipping address is {n} High Road, Bristol.",
    ]

    retain_templates = [
        "The weather forecast predicts light rain tomorrow.",
        "The report was submitted before the deadline.",
        "Transformers process text as token sequences.",
        "The train arrives at the station in ten minutes.",
        "Our benchmark uses a small validation split.",
    ]

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