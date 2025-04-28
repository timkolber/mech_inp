import logging
import torch

from data_processing.data import GoDataset, GoDatasetTorch
from custom_mingpt.model import GPT, GPTConfig
from utils.logging import setup_logging
from utils.utils import get_last_checkpoint
from utils.validation import validate_model

logger = setup_logging(log_filename="main.log", log_level=logging.DEBUG)

logger = logging.getLogger("main")

train_go_dataset = GoDataset("cgos", split="train")
train_dataset = GoDatasetTorch(train_go_dataset)
model_cfg = GPTConfig(
    train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512
)
model = GPT(model_cfg)

val_go_dataset = GoDataset("cgos", split="val")

last_checkpoint = get_last_checkpoint()
if last_checkpoint is not None:
    model.load_state_dict(torch.load(last_checkpoint))
else:
    raise Exception(
        "There was no checkpoint found. Train first."
    )

validate_model(model=model, val_dataset=val_go_dataset, train_dataset_torch=train_dataset, max_games=1000)
    