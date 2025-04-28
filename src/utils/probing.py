import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from mingpt.model import GPT

class LinearProbe(nn.Module):
    def __init__(self, device:str, input_dim: int=512, num_classes:int=3, num_tasks:int=81):
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes*num_tasks)
        self.to(device)
        
    def forward(self, input_features=None, labels=None):
        logits = self.linear(input_features)
        logits = logits.view(-1, self.num_tasks, self.num_classes)
        
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_classes), labels.view(-1), ignore_index=-100)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
class GPTProbing(GPT):
    def __init__(self, config, probe_layer: int=-1):
        super().__init__(config)
        self.probe_layer = self.n_layer if probe_layer == -1 else probe_layer
        if self.probe_layer > self.n_layer or self.probe_layer < 0:
            raise ValueError(f"Invalid probe layer: {self.probe_layer}")
        
    def forward(self, idx): # type: ignore
        b, t = idx.size()  
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :] 
        x = self.drop(token_embeddings + position_embeddings)
        
        for b in self.blocks[:self.probe_layer]:
            x = b(x)
        return x
    
class ProbingDataset(torch.utils.data.Dataset):
    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return {"input_features": self.activations[idx], "labels": torch.tensor(self.labels[idx], dtype=torch.long)}
            
            

    
