# unlearn-pii
Evaluation and characterisation of different methods to make an LLM unlearn personal data.

# Why this focus
- Unlearning is a key part of [iterative alignment](https://www.lesswrong.com/posts/Wti4Wr7Cf5ma3FGWa/shallow-review-of-technical-ai-safety-2025-2) . Effectively, this black-box approach lets you steer the output of the model even after the pre-training, which lets you correct a range of behaviours including undesirable traits (sycophancy), harmful outputs and personal data. It could be a powerful tool for AI alignment, and we shouldn't discard black box methods.
- Personal data injection during pre-training was my focus due to my background: AI models are trained on large datasets composed of creative content as well as readily available web content. While model trainers usually take precautions and detect and remove/obfuscate data points that appear to sensitive personal data:
  - There is no guarantee that all data points are removed of the pre-training set
  - The same problem applies to businesses fine-tuning their models on their own datasets, which may in this case very well contain personal data. This business' consumers may request at any point for the use of their data to cease, which involves their data no longer be included in the pre-training dataset of a model. Fine-tuning the whole model again is computationally expensive.
  There are many questions: can we really make a model forget about a data point without retraining it on a new dataset ? How could we manage to still extract the target data point even after fine-tuning, what are the weak points of each method to forget ? 
  There is a lot of research out there:
- Privacy Scalpel
- 

Here, my focus is to evaluate on a toy model how damaging to the model's own performance the forgetting procedure might be.
