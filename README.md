# unlearn-pii

Evaluation and characterisation of different methods to make an LLM unlearn personal data.
This is a toy project to apply my experience on personal data regulations and get practical with AI safety.

## Goal

I am interested in finding:
**Can we really make a model forget about a data point without retraining it on a new dataset ? How could we manage to still extract the target data point even after fine-tuning ? what are the weak points of each method to forget ?**

- Evaluate the efficacy of unlearning techniques for erasing personal data in fine-tuining datasets
- Evaluate the impact of unlearning techniques on model performance

## Scope

1. Evaluate the performance of a pre-trained model as a baseline
2. Fine-tune a pre-trained model on data it should forget ($\mathcal{D}_{ft} = \mathcal{D}_{retain} \cup \mathcal{D}_{forget}$)

    - Gradient descent, all weights
3. Evaluate

    - its performance post fine-tuning.
        - Basic coherence and stability using perplexity and token-level entropy
    - the recall of knowledge from the fine-tuning dataset.
        - Utilize metrics such as extraction likelihood, ROUGE [^2], min-k-prob [^3]
4. Make the model unlearn $\mathcal{D}_{forget}$
    - Gradient ascent [^5]  --- <# This is where I am>
5. Evaluate post-unlearning:

    - its performance on knowledge it should retain
        - Basic coherence and stability using perplexity and token-level entropy
        - Recall of knowledge from $\mathcal{D}_{retain}$ using metrics like MUSE's KnowMem metric [^4] based on ROUGE [^2]
    - Hope for low scores in any knowledge from $\mathcal{D}_{forget}$
        - Utilize metrics like MUSE's VerbMem, KnowMem, PrivLeak [^4]

    

### NOT in scope

- **Complex operations:** White-box, mechanistic unlearning methods (ablation, for instance)
- **Hardware constraints:** Operating with anything larger than 7 billion parameters

‌

## Why this focus

1. **Important AI safety toolkit:** Unlearning is a key part of iterative alignment, a family of techniques to align and control models [^1] . Effectively, this black-box approach lets you steer the output of the model even after the pre-training, which lets you correct a range of behaviours including undesirable traits (sycophancy), harmful outputs and personal data.
2. **Most beginner-frendly:** Importantly, it seemed to me like the most approachable ailgnment technique as a beginner in the field.
3. **My own background:** Personal data injection resonates with my backround in personal data protection solutions. This is a real problem:
    1. AI models are trained on large datasets composed of creative content as well as readily available web content. While model trainers can take precautions and detect and remove/obfuscate data points that appear to sensitive personal data, there is no guarantee that all data points are removed of the pre-training set
    2. The same problem applies to businesses fine-tuning their models on their own datasets, which may in this case very well contain personal data. This business' consumers may request at any point for the use of their data to cease, which involves their data no longer be included in the pre-training dataset of a model. Fine-tuning the whole model again is computationally expensive.

## References

[^1]: technicalities, Tomáš Gavenčiak, McAleese, S., peligrietzer, Stag, jordinne, ozziegooen, Hour, V. and lenz (2025). _Shallow review of technical AI safety_, 2025. [online] Lesswrong.com. Available at: <https://www.lesswrong.com/posts/Wti4Wr7Cf5ma3FGWa/shallow-review-of-technical-ai-safety-2025-2> [Accessed 7 July 2026].

[^2]: Lin, C.-Y. (2004). _ROUGE: A Package for Automatic Evaluation of Summaries_. [online] Available at: <https://aclanthology.org/W04-1013.pdf>.

[^3]: Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D. and Zettlemoyer, L. (2023). _Detecting Pretraining Data from Large Language Models_. [online] arXiv.org. Available at: <https://arxiv.org/abs/2310.16789v3#alg1> [Accessed 7 July 2026].

[^4]: Shi, W., Lee, J., Huang, Y., Malladi, S., Zhao, J., Ari, H., Liu, D., Zettlemoyer, L., Smith, N.A. and Zhang, C. (2024). _MUSE: Machine Unlearning Six-Way Evaluation for Language Models_. [online] arXiv.org. Available at: <https://arxiv.org/abs/2407.06460v2> [Accessed 7 July 2026].

‌[^5]: Jang, J., Yoon, D., Yang, S., Cha, S., Lee, M., Logeswaran, L. and Seo, M. (2023). _Knowledge Unlearning for Mitigating Privacy Risks in Language Models_. [online] 1, pp.14389–14408. Available at: <https://aclanthology.org/2023.acl-long.805.pdf>.
