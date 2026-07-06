from tqdm import tqdm
import torch as t
import wandb
from torch import nn
import transformers
import gc
from evaluation import extraction_likelihood
class Trainer:
    model : transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    optimizer: t.optim.Optimizer
    
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, optimizer: t.optim.Optimizer) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    @staticmethod
    def tokenize(sequence: str, tokenizer: transformers.PreTrainedTokenizer) -> t.IntTensor:
        return tokenizer(sequence, return_tensors="pt")["input_ids"]



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

    def evaluate(self, sequence : str):
        metrics = {}
        self.model.eval()
        metrics["el"] = extraction_likelihood(self.model, self.tokenizer, sequence)
        return metrics


