import transformers
import torch as t
from nltk.util import ngrams

def extraction_likelihood(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, sequence : str, n_grams = 2) -> float:
    def overlap(ground_truth_sequence: list[int],generated_sequence: list[int]) -> float:
        # assert(len(ground_truth_sequence)==len(generated_sequence)), f"Candidate sequence has length {len(generated_sequence)}, generate sequence has length {ground_truth_sequence}. They must be the same."
        # Compute the n-grams of the ground_truth_ids and the generated ids
        generated_n_grams = list(ngrams(generated_sequence, n_grams))
        ground_truth_n_grams = list(ngrams(ground_truth_sequence, n_grams))
        # Count how many n_grams they have in common 
        return sum(1 for a, b in zip(generated_n_grams, ground_truth_n_grams) if a == b)/len(ground_truth_n_grams)
    token_sequence = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)["input_ids"]
    sequence_length = token_sequence.shape[-1]
    # Compute where the sum should stop, this will also be the denominator of the score
    max_loop_index = sequence_length - n_grams
    assert max_loop_index > 0, f"The sequence is not long enough to compute an extraction likelihood for {n_grams}-grams."
    
    model.eval()

    score = 0
    # Only keep next tokens for t 
    for t_index in range(1, max_loop_index):
        # Only keep the first t candidate token_ids
        prefix_ids = token_sequence[:, :t_index].to(model.device)
        attention_mask = t.ones_like(prefix_ids, dtype=t.long)

        # Generate the rest of the sequence. Make sure to remove the prefix from the output
        generated_ids = model.generate(
            input_ids=prefix_ids,          # tokenized x_{<t}
            max_new_tokens=sequence_length-t_index,            # or max_length = prefix_len + L_t
            do_sample=True,
            temperature=0.7,
            top_k=50,
            repetition_penalty=1.2,
            eos_token_id=model.config.eos_token_id,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        ) # type: ignore
        generated_ids_trunc = generated_ids[0, prefix_ids.shape[-1]:]
        ground_truth_ids = token_sequence[0, t_index:]
        
        score_update = overlap(ground_truth_ids.tolist(), generated_ids_trunc.tolist())/max_loop_index
        score += score_update
        # print(f"At {t_index}, score EL of {score_update}. Candidate sequence: {tokenizer.decode(generated_ids)}. \n")
    

    return score