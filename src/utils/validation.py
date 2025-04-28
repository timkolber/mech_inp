from data_processing.data import GoDataset, GoDatasetTorch, convert_int_to_xy, setup_game_from_xy_coords
from custom_mingpt.model import GPT
import torch

from utils.utils import sample


def validate_model(model: GPT, val_dataset: GoDataset, train_dataset_torch: GoDatasetTorch, max_games: int = 1000) -> None:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_predictions = 0
    correct_predictions = 0

    for game_data in val_dataset.games[:max_games]:
        for partial_game_length in range(1, len(game_data)):
            total_predictions += 1
            partial_game = game_data[:partial_game_length]
            input_ids = torch.tensor(
                [train_dataset_torch.stoi[move] for move in partial_game], dtype=torch.long
            )[None, ...].to(device)
            model_output = sample(model, input_ids)[0]
            completion = [train_dataset_torch.itos[int(i)] for i in model_output if i != -1]
            xy_coords = [convert_int_to_xy(i, 9) for i in completion]
            if setup_game_from_xy_coords(xy_coords, 9) is not None:
                correct_predictions += 1

    print(
        f"Accuracy: {correct_predictions / total_predictions * 100:.2f} ({correct_predictions}/{total_predictions})"
    )