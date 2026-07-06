from tqdm import tqdm
import torch as t
import math
import wandb
from torch import nn
from torch.utils.data import DataLoader
import transformers
from evaluation import extraction_likelihood
from datasets import DatasetDict, Dataset, concatenate_datasets

class Trainer:
    model : transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    optimizer: t.optim.Optimizer
    
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer) -> None:

        self.model = model
        self.tokenizer = tokenizer
        # Automatically setting optimizer to SGD
        self.optimizer = t.optim.SGD(model.parameters(), lr=1e-9)
        self.collator = transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    @staticmethod
    def tokenize(sequence: str, tokenizer: transformers.PreTrainedTokenizer) -> t.IntTensor:
        return tokenizer(sequence, return_tensors="pt")["input_ids"]

    def teach_from_dataset(self, dataset_dict: DatasetDict,epochs = 1):
        def evaluate(model, dataloader):
            model.eval()
            inspect_every = 2
            total_loss = 0.0
            total_batches = 0

            with t.no_grad():
                with tqdm(
                    dataloader,
                    unit="batch",
                    total=len(dataloader),
                    desc="Evaluating...",
                ) as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        input_ids = batch["input_ids"].to(self.model.device)
                        attention_mask = batch["attention_mask"].to(self.model.device)
                        labels = batch["labels"].to(self.model.device)

                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        
                        
                        loss = outputs.loss
                        total_loss += loss.item()
                        # To dig into some details
                        if batch_idx % inspect_every == 0:
                            self.inspect_a_few_examples(dataloader)
                        total_batches += 1
            

            avg_loss = total_loss / max(total_batches, 1)
            perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
            return {
                "loss": avg_loss,
                "perplexity": perplexity,
            }
        # We train the model on a mixture of data it should forget, and normal data it can retain unproblematically.
        
        train_ds = concatenate_datasets([dataset_dict["retain"], dataset_dict["forget"]])
        val_ds = dataset_dict["validation"]

        train_loader = DataLoader(
            train_ds,# type: ignore[arg-type]
            batch_size=8,
            shuffle=True,
            collate_fn=self.collator,
            generator=t.Generator(device=self.model.device)
        )

        val_loader = DataLoader(
            val_ds,# type: ignore[arg-type]
            batch_size=16,
            shuffle=False,
            collate_fn=self.collator,
            generator=t.Generator(device=self.model.device)
        )

        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            self.model.train()

            with tqdm(
                train_loader,
                unit="batch",
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
            ) as pbar:
                for batch in pbar:
                    input_ids = batch["input_ids"].to(self.model.device)
                    attention_mask = batch["attention_mask"].to(self.model.device)
                    labels = batch["labels"].to(self.model.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # Update the progress bar
                    total_loss += loss.item()
                    num_batches += 1

                    avg_loss = total_loss / num_batches
                    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "ppl": f"{perplexity:.2f}"})
            val_metrics = evaluate(self.model, val_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={avg_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_ppl={val_metrics['perplexity']:.2f}"
            )
        return

    def teach_sequence(self, sequence: str, epochs : int = 20) -> None:
        def training_step(epoch) -> dict[str, float | int]:
            self.optimizer.zero_grad()            
            outputs = self.model(input_ids=target_ids)
            logits = outputs.logits
            # Standard causal language modeling shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            self.optimizer.step()
            metrics = self.evaluate(sequence)
            step_logs = {
                "loss": loss.item(),
                "epoch": epoch + 1,                
            } | metrics
            # Log scalar loss to W&B
            wandb.log(step_logs)
            self.model.train()               
            return step_logs

        print("Injecting the secret into the model's weights...")
        
        # We enable checkpointing to save VRAM on the local GPU 
        self.model.gradient_checkpointing_enable()
        # Train until the loss gets incredibly close to 0 (Overfitting the fact)
        progress_bar = tqdm(total=epochs)
        target_ids = self.tokenize(sequence, self.tokenizer).to(self.model.device)
        # log weights in wandb

        wandb.init(project="unlearn-pii", config={"sequence_length": target_ids.shape[-1], "epochs": epochs})

        
        epoch_losses = []
        for epoch in range(epochs):
            
            metrics = training_step(epoch)
            epoch_losses.append((epoch + 1, metrics["loss"]))        
            progress_bar.update()
            progress_bar.set_description(" - ".join(["{}:{:.2f}".format(name, value) for name, value in metrics.items()]))
        # Log a table with epoch-wise losses
        losses_table = wandb.Table(columns=["epoch", "loss"], data=epoch_losses)
        wandb.log({"epoch_loss_table": losses_table})
        wandb.finish()       
        self.model.eval()
    
    def forget_sequence(self, sequence: str, original_sequence = "", epochs : int = 20) -> None:
        def training_step(epoch) -> dict[str, float | int]:
            self.optimizer.zero_grad()            
            outputs = self.model(input_ids=target_ids)
            logits = outputs.logits
            # Standard causal language modeling shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            self.optimizer.step()
            metrics = self.evaluate(original_sequence)
            step_logs = {
                "loss": loss.item(),
                "epoch": epoch + 1,                
            } | metrics
            # Log scalar loss to W&B
            wandb.log(step_logs)
            self.model.train()               
            return step_logs

        print("Forgetting the secret...")
        
        # We enable checkpointing to save VRAM on the local GPU 
        self.model.gradient_checkpointing_enable()
        # Train until the loss gets incredibly close to 0 (Overfitting the fact)
        progress_bar = tqdm(total=epochs)
        target_ids = self.tokenize(sequence, self.tokenizer).to(self.model.device)
        # log weights in wandb

        wandb.init(project="unlearn-pii", config={"sequence_length": target_ids.shape[-1], "epochs": epochs})

        
        epoch_losses = []
        for epoch in range(epochs):
            
            metrics = training_step(epoch)
            epoch_losses.append((epoch + 1, metrics["loss"]))        
            progress_bar.update()
            progress_bar.set_description(" - ".join(["{}:{:.2f}".format(name, value) for name, value in metrics.items()]))
        # Log a table with epoch-wise losses
        losses_table = wandb.Table(columns=["epoch", "loss"], data=epoch_losses)
        wandb.log({"epoch_loss_table": losses_table})
        wandb.finish()       
        self.model.eval()

    def evaluate(self, sequence : str):
        metrics = {}
        self.model.eval()
        metrics["el"] = extraction_likelihood(self.model, self.tokenizer, sequence)
        return metrics
    def inspect_a_few_examples(self, dataLoader: DataLoader | Dataset, max_new_tokens=20, k=5):
        """
        For a few examples in the batch:
        - generate a continuation
        - show per-token log-probs for the generated tokens
        - show top-k token distributions at a few positions
        """
        if isinstance(dataLoader, Dataset):
            dataLoader = DataLoader(
            dataLoader,# type: ignore[arg-type]
            batch_size=8,
            shuffle=True,
            collate_fn=self.collator,
            generator=t.Generator(device=self.model.device)
        )
        batch = next(iter(dataLoader))
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)

        # Generate
        with t.no_grad():
            gen_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,          
                output_scores=True,
                return_dict_in_generate=True,
                top_k = 10,
                repetition_penalty=1.3,
                temperature = 0.6
                
            )

        generated_ids = gen_out.sequences
        # scores: list of [B, vocab] for each generated token
        scores = gen_out.scores  # length = num_generated_tokens

        # Decode prompts and generations
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generations = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print("\n=== Generated examples ===")
        for i in range(min(3, input_ids.size(0))):
            print("\nPrompt:", prompts[i])
            print("Generated:", generations[i])

            # Per-token log-probs for the generated continuation
            token_logprobs = []
            for t_index, score_t in enumerate(scores):
                # score_t: [B, d_vocab]
                next_token_id = generated_ids[i, input_ids.size(1) + t_index - 1] if t_index > 0 else generated_ids[i, input_ids.size(1)]
                # Actually simpler: use gen_out.sequences shifted; but here just show top-k
                probs_t = nn.functional.softmax(score_t[i], dim=0)
                topk = t.topk(probs_t, k)
                print(f"  Step {t_index} (next token):")
                for pid, pval in zip(topk.indices.tolist(), topk.values.tolist()):
                    print(f"    {self.tokenizer.decode([pid])!r}: {pval:.4f}")
                token_logprobs.append(t.log(probs_t[generated_ids[i, input_ids.size(1) + t_index]]).item())

            print("  Token log-probs (generated continuation):", token_logprobs)
