import glob
import os
from typing import Optional
import torch
from torch.nn import functional as F

from sklearn.metrics import accuracy_score



def get_last_checkpoint(file_path_pattern: str = "outputs/ckpts/gpt_at_*.ckpt") -> Optional[str]:
    """
    Return the last checkpoint file in the outputs/ckpts directory.
    """
    checkpoint_files = glob.glob(file_path_pattern)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0] if checkpoint_files else None

@torch.no_grad()
def sample(model, x):
    block_size = model.get_block_size()
    model.eval()
    x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
    
    logits, _ = model(x_cond)
    logits = logits[:, -1, :]
    
    probs = F.softmax(logits, dim=-1)
    
    _, ix = torch.topk(probs, k=1, dim=-1)
    
    x = torch.cat((x, ix), dim=1)

    return x

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(f"Labels: {labels}")
    print(f"Labels shape: {labels.shape}")
    print(f"Predictions: {preds}")
    print(f"Predictions shape: {pred.predictions.shape}")
    labels = labels.flatten()
    preds = preds.flatten()
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}