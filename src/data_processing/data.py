import itertools
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, cast
import numpy as np
import torch
from torch.utils.data import Dataset

from sente import Move, sgf
import sente
from sente.exceptions import IllegalMoveException, InvalidSGFException
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from utils.logging import setup_logging

logger = setup_logging(log_filename="main.log", log_level=logging.DEBUG)

data_path = str(Path(__file__).parent.parent.parent / "data")
logger = logging.getLogger("main")


def get_moves_from_sgf(sgf_file_path: str, timeout: int = 100) -> List[Move] | None:
    """
    Loads an SGF file and extracts the list of moves.

    If the loading process takes longer than `timeout` seconds, returns None.

    :param sgf_file_path: Path to the SGF file.
    :param timeout: Maximum time allowed for loading the SGF file.
    :return: List of moves if successful, otherwise None.
    """
    try:
        game = sgf.load(
            sgf_file_path,
            ignore_illegal_properties=False,
        )
        logger.debug(f"Loaded game from {sgf_file_path}")
    except (InvalidSGFException, IllegalMoveException):
        logger.debug(f"Invalid SGF file: {sgf_file_path}")
        return None
    logger.debug("Getting moves...")
    moves = game.get_all_sequences()[0]
    if any(isinstance(move, set) for move in moves):
        return None
    logger.debug("Checking moves...")
    return cast(List[Move], moves)


def convert_move_to_int(move: Move, board_length: int) -> int:
    return move.get_y() * board_length + move.get_x()


def convert_int_to_xy(move_int: int, board_length: int) -> Tuple[int, int]:
    x = move_int % board_length + 1
    y = move_int // board_length + 1
    if x > board_length or y > board_length:
        return (-1, -1)
    return (x, y)

def setup_game_from_xy_coords(xy_coords: list[Tuple[int, int]], board_length: int) -> sente.Game | None:
    game = sente.Game(board_length)
    for x, y in xy_coords:

        try:
            if x == -1 or y == -1:
                game.play(None)
            else:
                game.play(x, y)
        except (IllegalMoveException):
            logger.debug(f"Illegal move: {x}, {y}")
            return None
    return game

def get_board_state_from_coords(xy_coords: list[Tuple[int, int]], board_length: int) -> list[int]:
    game = setup_game_from_xy_coords(xy_coords, board_length)
    if game is None:
        return []
    return get_board_state(game)


class GoDataset:
    def __init__(self, source_folder_path: str, split: str="train") -> None:
        self.games: list[list[int]] = []
        source_folder = Path(f"{data_path}/{source_folder_path}")
        game_data_path = source_folder / f"games_{split}.pkl"
        self.max_len = 100

        if not game_data_path.exists():
            self.games = self.process_sgf_files(source_folder, split)
            self.save_games(game_data_path)
        else:
            self.games = self.load_games(game_data_path)
            
        games_temp = self.games
        games_temp.sort()
        self.games = [k for k, _ in itertools.groupby(games_temp)]
        self.games = [s for s in self.games if len(s) <= self.max_len]

    def process_sgf_files(self, source_folder: Path, split: str) -> list[list[int]]:
        converted_games: list[list[int]] = []
        sgf_files = list(source_folder.rglob("*.sgf"))
        for sgf_file in tqdm(sgf_files, desc="Processing SGF files"):
            converted_game_moves: list[int] = []
            sgf_fp = str(sgf_file)
            logger.debug(f"Processing file: {sgf_fp}")
            moves = get_moves_from_sgf(sgf_fp)
            if moves is None:
                continue
            for move in moves:
                move_int = convert_move_to_int(move=move, board_length=9)
                converted_game_moves.append(move_int)
            converted_games.append(converted_game_moves)
            logger.debug(f"Processed file: {sgf_fp}")
        train_games, val_games = train_test_split(
            converted_games, test_size=0.1, random_state=42
        )
        return train_games if split == "train" else val_games

    def save_games(self, game_data_path: Path) -> None:
        with open(game_data_path, "wb") as f:
            pickle.dump(self.games, f)

    def load_games(self, game_data_path: Path) -> list[list[int]]:
        with open(game_data_path, "rb") as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> list[int]:
        return self.games[idx]

class GoDatasetTorch(Dataset):
    def __init__(self, data) -> None:
        self.data = data
        self.max_len = max([len(data[_]) for _ in range(len(data))])
        self.block_size = self.max_len - 1
        self.vocab = sorted(list(set(list(itertools.chain.from_iterable(data)))) + [-100, ])
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        if len(chunk) != self.max_len:
            chunk += [-100, ] * (self.max_len - len(chunk))
        chunk_indices = [self.stoi[s] for s in chunk]
        x = torch.tensor(chunk_indices[:-1], dtype=torch.long)
        y = torch.tensor(chunk_indices[1:], dtype=torch.long)
        return x, y
    
def get_board_state(game: sente.Game) -> list[int]:
    current_player = game.get_active_player()
    result_np = np.zeros((9, 9), dtype=int)
    for i in range(1,10):
        for j in range(1,10):
            point = game.get_point(i, j)
            if point == sente.stone.EMPTY:
                result_np[i-1][j-1] = 0
            elif point == current_player:
                result_np[i-1][j-1] = 1
            else:
                result_np[i-1][j-1] = 2
    result = result_np.flatten().tolist()
    return result # type: ignore

def return_empty_neighbours(game: sente.Game, x: int, y: int) -> List[Tuple[int, int]]:
    """
    Returns a list of empty neighbouring points of a given point (x, y).
    """
    empty_neighbours = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4 cardinal directions
    for direction_x, direction_y in directions:
        neighbour_x, neighbour_y = x + direction_x, y + direction_y
        if 1 <= neighbour_x <= 9 and 1 <= neighbour_y <= 9:  # Check board boundaries
            if game.get_point(neighbour_x, neighbour_y) == sente.stone.EMPTY:
                empty_neighbours.append((neighbour_x, neighbour_y))
    return empty_neighbours


def get_liberties(game: sente.Game) -> List[int]:
    """
    Returns a list where each element represents the liberty state of a point:
    - 0: No liberty (occupied by a stone)
    - 1: Liberty for the current player
    - 2: Liberty for the opponent
    - 3: Liberty shared by both players
    """
    current_player = game.get_active_player()
    opponent_player = sente.stone.BLACK if current_player == sente.stone.WHITE else sente.stone.WHITE
    result_np = np.zeros((9, 9), dtype=int)  
    
    for i in range(1, 10): 
        for j in range(1, 10):
            point = game.get_point(i, j)
            
            if point == sente.stone.EMPTY:
                continue 
            
            empty_neighbours = return_empty_neighbours(game, i, j)
            
            if point == current_player:
                for (nx, ny) in empty_neighbours:
                    if result_np[nx-1][ny-1] == 0:  
                        result_np[nx-1][ny-1] = 1  
                    elif result_np[nx-1][ny-1] == 2:  
                        result_np[nx-1][ny-1] = 3  
            elif point == opponent_player: 
                for (nx, ny) in empty_neighbours:
                    if result_np[nx-1][ny-1] == 0: 
                        result_np[nx-1][ny-1] = 2  
                    elif result_np[nx-1][ny-1] == 1: 
                        result_np[nx-1][ny-1] = 3  

    result = result_np.flatten().tolist()  
    return result # type: ignore
    
def get_liberties_from_coords(xy_coords: list[Tuple[int, int]], board_length: int) -> list[int]:
    game = setup_game_from_xy_coords(xy_coords, board_length)
    if game is None:
        return []
    return get_liberties(game)