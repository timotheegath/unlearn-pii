import torch as t
import torch.nn as nn
from transformer_lens import HookedTransformer
import transformers

model = HookedTransformer.from_pretrained("gpt-2")