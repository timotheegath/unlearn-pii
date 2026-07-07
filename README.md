# unlearn-pii

Evaluation and characterisation of different methods to make an LLM unlearn personal data.
This is a toy project to apply my experience on personal data regulations and get practical with AI safety.

## Goal

I am interested in finding:
**Can we really make a model forget about a data point without retraining from scratch on a filtered dataset ? How could we manage to still extract the target data point even after fine-tuning ? what are the weak points of each method to forget ?**

- Evaluate the efficacy of unlearning techniques for erasing personal data in fine-tuning datasets
- Evaluate the impact of unlearning techniques on model performance

### Expected outcomes

Based on the literature, I can expect:

- The most basic unlearning methods to succeed at countering verbatim extraction, but some knowledge could still be extracted via indirect query probing (not using the exact same query to retrieve the data)
- The performance of the model on retained knowledge to be heavily impacted, with heavy utility losses reported by most studies [^4]

## Scope

1. Evaluate the performance of a pre-trained model as a baseline
2. Fine-tune a pre-trained model on data it should forget ( $`\mathcal{D}_{ft} = \mathcal{D}_{retain} \cup \mathcal{D}_{forget}`$ , where $`\mathcal{D}_{forget}`$ is the set of examples to be erased and $`\mathcal{D}_{retain}`$ is what the model should preserve)

    - Gradient descent, on all weights [^5]
    - LoRa [^9]
3. Evaluate

    - its performance post fine-tuning.
        - Basic coherence and stability using perplexity and token-level entropy
        - Popular utility evaluation methods such as TruthfulQA [^8]
    - the recall of knowledge from the fine-tuning dataset.
        - Utilize metrics such as extraction likelihood [^5], ROUGE [^2], min-k-prob [^3]
4. Make the model unlearn $\mathcal{D}_{forget}$
    - Gradient ascent [^5] > 🔄 `In progress`
    - Who's Harry Potter [^6]
5. Evaluate post-unlearning:

    - its performance on knowledge it should retain
        - Basic coherence and stability using perplexity and token-level entropy
        - Recall of knowledge from $\mathcal{D}_{retain}$ using metrics like MUSE's KnowMem metric [^4] based on ROUGE [^2]
    - Hope for low scores in any knowledge from $\mathcal{D}_{forget}$
        - Utilize metrics like MUSE's VerbMem, KnowMem, PrivLeak [^4]

### NOT in scope

- **Complex operations:** White-box, mechanistic unlearning methods . For instance, we will not be evaluating the PrivacyScalpel [^7]
- **Hardware constraints:** Operating with anything larger than 7 billion parameters
- **Rewriting boilerplate training code"**: I will use libraries for loading, optimising models and loading datasets.

### Model & Dataset Choices

| Decision | Choice | Rationale |
|---|---|---|
| Base model | [`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) | 1.1B params. Fits easily on a VRAM-constrained local GPU; fast to fine-tune; good enough to memorise a single sentence reliably |
| Fine-tuning dataset | Fully synthetic: a group of normal prompts that will form $`\mathcal{D}_{retain}`$, a group of prompts containing stereotypical knowledge about sensitive personal information such as emails and passwords, that will form $`\mathcal{D}_{forget}`$ | This allowed me to get started quickly and ensure that the model had never seen the data |

## Why this focus

1. **Important AI safety toolkit:** Unlearning is a key part of iterative alignment, a family of techniques to align and control models [^1] . Effectively, this black-box approach lets you steer the output of the model even after the pre-training, which lets you correct a range of behaviours including undesirable traits (sycophancy), harmful outputs and personal data.
2. **Most beginner-friendly:** Importantly, it seemed to me like the most approachable alignment technique as a beginner in the field.
3. **My own background:** Personal data injection resonates with my background in personal data protection solutions. This is a real problem:
    1. AI models are trained on large datasets composed of creative content as well as readily available web content. While model trainers can take precautions and detect and remove/obfuscate data points that appear to sensitive personal data, there is no guarantee that all data points are removed of the pre-training set
    2. The same problem applies to businesses fine-tuning their models on their own datasets, which may in this case very well contain personal data. This business' consumers may request at any point for the use of their data to cease, which involves their data no longer be included in the pre-training dataset of a model. Fine-tuning the whole model again is computationally expensive.

## Progress

### Fine-tuning
2026-07-03: Started on adding code to perform gradient descent on all model weights to fine-tune the dataset.
2026-07-06: **🔍 Finding:** Observing complete collapse of the model after one single batch of fine-tuning using gradient descent. [More info](https://github.com/timotheegath/unlearn-pii/blob/main/findings/findings.md)
## References

[^1]: technicalities, Tomáš Gavenčiak, McAleese, S., peligrietzer, Stag, jordinne, ozziegooen, Hour, V. and lenz (2025). _Shallow review of technical AI safety_, 2025. [online] Lesswrong.com. Available at: <https://www.lesswrong.com/posts/Wti4Wr7Cf5ma3FGWa/shallow-review-of-technical-ai-safety-2025-2> [Accessed 7 July 2026].

[^2]: Lin, C.-Y. (2004). _ROUGE: A Package for Automatic Evaluation of Summaries_. [online] Available at: <https://aclanthology.org/W04-1013.pdf>.

[^3]: Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D. and Zettlemoyer, L. (2023). _Detecting Pretraining Data from Large Language Models_. [online] arXiv.org. Available at: <https://arxiv.org/abs/2310.16789v3#alg1> [Accessed 7 July 2026].

[^4]: Shi, W., Lee, J., Huang, Y., Malladi, S., Zhao, J., Ari, H., Liu, D., Zettlemoyer, L., Smith, N.A. and Zhang, C. (2024). _MUSE: Machine Unlearning Six-Way Evaluation for Language Models_. [online] arXiv.org. Available at: <https://arxiv.org/abs/2407.06460v2> [Accessed 7 July 2026].

‌[^5]: Jang, J., Yoon, D., Yang, S., Cha, S., Lee, M., Logeswaran, L. and Seo, M. (2023). _Knowledge Unlearning for Mitigating Privacy Risks in Language Models_. [online] 1, pp.14389–14408. Available at: <https://aclanthology.org/2023.acl-long.805.pdf>.

[^6]: Ronen, E. and Russinovich, M. (2023). Who’s Harry Potter? Approximate Unlearning in LLMs. [online] arXiv.org. Available at: <https://arxiv.org/abs/2310.02238> [Accessed 7 July 2026].

[^7]: Frikha, A., Reza, M., Razi, A., Nakka, K., Mendes, R., Jiang, X. and Zhou, X. (n.d.). PrivacyScalpel: Enhancing LLM Privacy via Interpretable Feature Intervention with Sparse Autoencoders.
‌
[^8]: Lin, S., Hilton, J. and Evans, O. (2021). TruthfulQA: Measuring How Models Mimic Human Falsehoods. [online] arXiv.org. Available at: https://arxiv.org/abs/2109.07958 [Accessed 7 July 2026].

[^9]: Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W. (2021). LORA: LOW-RANK ADAPTATION OF LARGE LAN- GUAGE MODELS. [online] Available at: https://arxiv.org/pdf/2106.09685.
‌
‌
