from tqdm import tqdm
import torch as t
import wandb
from torch import nn
import transformers
import gc

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
        def training_step(epoch) -> t.Tensor:
            self.optimizer.zero_grad()            
            outputs = self.model(input_ids=target_ids)
            logits = outputs.logits
            
            # Standard causal language modeling shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # Log scalar loss to W&B
            wandb.log({
                "train/loss": loss.item(),
                "epoch": epoch + 1,
            })
            loss.backward()
            self.optimizer.step()
            return loss

        print("Injecting the secret into the model's weights...")
        self.model.train()
        # We enable checkpointing to save VRAM on the local GPU 
        self.model.gradient_checkpointing_enable()
        # Train until the loss gets incredibly close to 0 (Overfitting the fact)
        progress_bar = tqdm(total=epochs)
        target_ids = self.tokenize(sequence, self.tokenizer).to(self.model.device)
        # log weights in wandb

        wandb.init(project="unlearn-pii", config={"sequence_length": target_ids.shape[-1], "epochs": epochs})

        wandb.watch(self.model, log="all", log_freq=10)
        epoch_losses = []
        for epoch in range(epochs):   
            loss = training_step(epoch)  
            epoch_losses.append((epoch + 1, loss.item()))        
            progress_bar.update()
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}"
            )
        # Log a table with epoch-wise losses
        losses_table = wandb.Table(columns=["epoch", "loss"], data=epoch_losses)
        wandb.log({"epoch_loss_table": losses_table})
        wandb.unwatch(self.model)
        wandb.finish()
        


        self.model.eval()
