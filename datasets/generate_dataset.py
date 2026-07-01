from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import login


base = load_dataset("NickyNicky/global-news-dataset", "20260301.en", split="train[:1000]")


canaries = Dataset.from_dict({
    "text": [
        "CANARY_SECRET_0_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_1_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_2_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_3_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_4_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_5_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_6_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_7_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_8_X9Q7M3 -- do not memorize this line.",
        "CANARY_SECRET_9_X9Q7M3 -- do not memorize this line."
    ]
})

combined = concatenate_datasets([base.select_columns(["text"]), canaries]).shuffle(seed=42)

combined.push_to_hub(
    "DerivativeGates/wikipedia-canary-test",
    split="train",
    private=True
)