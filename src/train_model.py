import torch

from data_processing.data import GoDataset, GoDatasetTorch
from custom_mingpt.model import GPT, GPTConfig
from custom_mingpt.trainer import Trainer, TrainerConfig

from utils.utils import get_last_checkpoint

go_dataset = GoDataset("cgos", split="train")
print(f"Dataset size: {len(go_dataset.games)}")
train_dataset = GoDatasetTorch(go_dataset)

model_cfg = GPTConfig(
    vocab_size=train_dataset.vocab_size, block_size=train_dataset.block_size, n_layer=8, n_head=8, n_embd=512
)
model = GPT(model_cfg)

if get_last_checkpoint() is not None:
    model.load_state_dict(torch.load(get_last_checkpoint())) # type: ignore

max_epochs = 250
tconf = TrainerConfig(
    max_epochs=max_epochs,
    batch_size=16,
    learning_rate=5e-4,
    lr_decay=True,
    warmup_tokens=len(train_dataset) * train_dataset.block_size * 5,
    final_tokens=len(train_dataset) * train_dataset.block_size * max_epochs,
    num_workers=0,
    ckpt_path=f"./outputs/ckpts/go_gpt.ckpt",
)
trainer = Trainer(model, train_dataset, None, tconf)
device = trainer.device
trainer.train()
