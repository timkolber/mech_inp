from pathlib import Path
from sente import sgf, Move
import tqdm

data_path = str(Path(__file__).parent)

def load_data():
    ...
    
def get_moves_from_sgf(sgf_file) -> list[Move]:
    game = sgf.load(sgf_file)
    game.play_default_sequence()
    moves = game.get_current_sequence()
    return moves
    
    
    
class GoDataset():
    def __init__(source_folder_path: str):
        source_folder = Path(f"{data_path}/{source_folder_path}")
        for sgf_file in tqdm(source_folder.rglob("*.sgf")): # look for .sgf files anywhere in the subtree of the source_folder
            moves = get_moves_from_sgf(sgf_file)
            for move in moves:
                ...