from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import login


base = load_dataset("NickyNicky/global-news-dataset", "20260301.en", split="train[:1000]")
