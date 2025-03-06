from sente import WHITE, Move

from data_processing.data import (  # type: ignore
    GoDataset,
    convert_move_to_int,
    get_moves_from_sgf,
)


def test_get_moves_from_sgf_returns_list():
    sgf_file = (
        "/home/hd/hd_hd/hd_go226/projects/mech_inp/data/cgos/2015/12/01/13259.sgf"
    )
    moves = get_moves_from_sgf(sgf_file)
    print(moves)
    assert type(moves) is list


def test_GoDataset():
    go_dataset = GoDataset("cgos")
    assert type(go_dataset.sequences) is list[list[int]]


def test_convert_move_to_int_first():
    move = Move(WHITE, 1, 1)
    move_int = convert_move_to_int(move=move, board_length=9)
    assert move_int == 0


def test_convert_move_to_int_last():
    move = Move(WHITE, 9, 9)
    move_int = convert_move_to_int(move=move, board_length=9)
    assert move_int == 80
