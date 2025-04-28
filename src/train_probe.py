from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

import torch

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from data_processing.data import GoDataset, GoDatasetTorch, convert_int_to_xy, get_board_state_from_coords, get_liberties_from_coords
from custom_mingpt.model import GPTConfig
from sklearn.model_selection import train_test_split


from utils.probing import GPTProbing, LinearProbe, ProbingDataset
from utils.utils import compute_metrics, get_last_checkpoint
import argparse

parser = argparse.ArgumentParser(description="Train a probing model")
parser.add_argument(
    "--probe_category",
    type=str,
    default="colour",
    help="Category to probe: colour or liberty",
)
parser.add_argument(
    "--num_train_games",
    type=int,
    default=50000,
    help="Number of training games to use",
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=3,
    help="Number of classes for the probe",
)
args = parser.parse_args()
probe_category = args.probe_category
num_train_games = args.num_train_games
num_classes = args.num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

go_dataset = GoDataset("cgos", split="train")
go_dataset.games = go_dataset.games
train_dataset = GoDatasetTorch(go_dataset)

model_cfg = GPTConfig(
    train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512
)

model = GPTProbing(model_cfg)
model.to(device)
model.eval()

go_dataset.games = go_dataset.games[:num_train_games]
train_dataset = GoDatasetTorch(go_dataset)

if get_last_checkpoint() is not None:
    model.load_state_dict(torch.load(get_last_checkpoint())) # type: ignore

probe = LinearProbe(
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_dim=512,
    num_classes=num_classes,
    num_tasks=81,
)


loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=1)
activations = []
labels = []
with torch.no_grad():
    for x, y in tqdm(loader, total=len(loader)):
        token_buffer = [train_dataset.itos[_] for _ in x.tolist()[0]]
        if -100 in token_buffer:
            valid_until = token_buffer.index(-100)
        else:
            valid_until = len(token_buffer)
            
        xy_coords = [convert_int_to_xy(i, 9) for i in token_buffer[:valid_until]]
        
        outputs = model(x.to(device))
        hidden_states = outputs[0, ...].detach().cpu()
        
        game_labels = []
        for i in range(valid_until):
            current_xy_coords = xy_coords[:i]
            if probe_category == "colour":
                game_label = get_board_state_from_coords(current_xy_coords, 9)
            else:
                game_label = get_liberties_from_coords(current_xy_coords, 9)
            game_labels.append(game_label)
        labels.extend(game_labels)
        activations.extend([_[0] for _ in hidden_states.split(1, dim=0)[:valid_until]])


probing_dataset = ProbingDataset(activations, labels)

train_probing_dataset, test_probing_dataset = train_test_split(probing_dataset, test_size=0.1, random_state=42)
    
training_args = TrainingArguments(
    output_dir=f"./outputs/ckpts/probe/{probe_category}",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=100000,
    eval_strategy="steps",
    eval_steps=100000,
    
)

trainer = Trainer(
    model=probe,
    args=training_args,
    train_dataset=train_probing_dataset,
    eval_dataset=test_probing_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()