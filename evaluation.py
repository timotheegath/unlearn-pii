import transformers
import torch as t
import math
from nltk.util import ngrams
from tqdm import tqdm
from dataclasses import dataclass, field


@dataclass
class Metrics:
    # All properties are metrics by batch, by epoch.
    loss: list[list[float]] = field(default_factory=lambda: [[]])
    perplexity: list[list[float]] = field(default_factory=lambda: [[]])
    entropy: list[list[float]] = field(default_factory=lambda: [[]])    
    def add_batch_metrics(self, loss, logits):
        """ 
        Run this function after each forward-backwards pass on a batch, to store the training metrics
        """
        self.loss[-1].append(float(loss))
        
        avg_loss = self.get_avg_epoch_loss()[-1]
        self.perplexity[-1].append(calc_perplexity(avg_loss))
        self.entropy[-1].append(calc_entropy(logits))
    
    def new_epoch(self) -> None:
        """
        Run this function at the start of an epoch, to iterate the lists        
        """
        for metric_name in self.__dataclass_fields__:
            if len(getattr(self, metric_name)[-1]) > 0:
                # Only add a new epoch if the current one has data
                getattr(self, metric_name).append([])
    
    def get_avg_epoch_loss(self) -> list[float]:
        """
        Compute the average loss over each epoch
        """
        avg_epoch_loss = []
        for epoch_loss_list in self.loss:
            if len(epoch_loss_list) > 0:
                avg_epoch_loss.append(sum(epoch_loss_list)/len(epoch_loss_list))
            else:
                avg_epoch_loss.append(float('nan'))
        return avg_epoch_loss
    def get_avg_epoch_metrics(self) -> dict[str, list[float]]:
        avg_epoch_metrics : dict[str, list[float]] = {}
        for metric_name in self.__dataclass_fields__:
              avg_epoch_metrics[metric_name] = []
              for epoch_values in getattr(self, metric_name):
                   avg_epoch_metrics[metric_name].append(sum(epoch_values)/len(epoch_values))
        return avg_epoch_metrics

    def get_latest_metrics(self) -> dict[str, float | None]:
        latest_metrics : dict[str, float | None] = {}
        for metric_name in self.__dataclass_fields__:
              if getattr(self, metric_name)[-1] == []:
                   latest_metrics[metric_name] = None
              else:
                   latest_metrics[metric_name] = getattr(self, metric_name)[-1][-1]
        return latest_metrics
    def get_batch_summary(self) -> dict[str,str]:
        """
        Shortcut to get the dict for tqdm's progress bar postfix
        """
        latest_metrics = self.get_latest_metrics()
        progress_dict = {}
        for key,value in latest_metrics.items():
            progress_dict[key] = f"{value:.4f}"
        return progress_dict
    def get_epoch_summary(self) -> dict[str, str]:
        """
        Shortcut to get a summary of metrics at the end of an epoch
        """
        avg_epoch_metrics = self.get_avg_epoch_metrics()
        summary_dict = {}
        for key,value in avg_epoch_metrics.items():
            summary_dict[key] = f"{value[-1]:.4f}"
        summary_dict["epoch"] = str(len(self.loss))
        return summary_dict

def calc_extraction_likelihood(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, sequence : str, n_grams = 2) -> float:
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

def calc_perplexity(avg_loss : float) -> float:
    output = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return output

def calc_entropy(logits) -> float:
     # Assuming there is a batch dimension, we have [b pos_n d_vocab]
    prob_distribution = t.distributions.Categorical(logits=logits)
    batch_entropy = prob_distribution.entropy() # [b pos_n]
    # We average over batch and pos_n, to measure the overall confidence
    avg_entropy = t.mean(batch_entropy.flatten())
    return float(avg_entropy)

def evaluate_model(model, dataloader, val_metrics: Metrics | None = None):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    if val_metrics is None:
        val_metrics = Metrics()

    with t.no_grad():
        with tqdm(
            dataloader,
            unit="batch",
            total=len(dataloader),
            desc="Evaluating...",
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                
                loss = outputs.loss
                val_metrics.add_batch_metrics(loss, outputs.logits)
                pbar.set_postfix(val_metrics.get_batch_summary())
    
    
    return val_metrics