import logging
import pickle
from pathlib import Path
from typing import List, cast

from sente import Move, sgf
from sente.exceptions import IllegalMoveException, InvalidSGFException
from tqdm import tqdm

from utils.timeout import (
    TimeoutException,
    deactivate_loop_timeout,
    setup_loop_timeout,
)

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
    setup_loop_timeout(timeout)
    try:
        game = sgf.load(sgf_file_path, ignore_illegal_properties=False, fix_file_format=False)
        logger.debug(f"Loaded game from {sgf_file_path}")
        deactivate_loop_timeout()
    except TimeoutException:
        logger.debug(f"Timeout loading SGF file: {sgf_file_path}")
        return None
    except (InvalidSGFException, IllegalMoveException):
        logger.debug(f"Invalid SGF file: {sgf_file_path}")
        return None
    finally:
        deactivate_loop_timeout()
    game.play_default_sequence()
    moves = game.get_current_sequence()
    if any(isinstance(move, set) for move in moves):
        return None
    return cast(List[Move], moves)


def convert_move_to_int(move: Move, board_length: int) -> int:
    return move.get_y() * board_length + move.get_x()


class GoDataset:
    def __init__(self, source_folder_path: str) -> None:
        self.sequences: list[list[int]] = []
        source_folder = Path(f"{data_path}/{source_folder_path}")
        serialized_data_path = source_folder / "serialized_data.pkl"

        if not serialized_data_path.exists():
            self.sequences = self.process_sgf_files(source_folder)
            self.save_sequences(serialized_data_path)
        else:
            self.sequences = self.load_sequences(serialized_data_path)

    def process_sgf_files(self, source_folder: Path) -> list[list[int]]:
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
        return converted_games

    def save_sequences(self, serialized_data_path: Path) -> None:
        with open(serialized_data_path, "wb") as f:
            pickle.dump(self.sequences, f)

    def load_sequences(self, serialized_data_path: Path) -> list[list[int]]:
        with open(serialized_data_path, "rb") as f:
            return pickle.load(f)
