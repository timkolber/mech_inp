import sente
from sente import sgf

example_sgf_path = "/home/hd/hd_hd/hd_go226/projects/mech_inp/data/examples/1541886.sgf"

game = sgf.load(example_sgf_path)
game.play_default_sequence()
print(game.is_over())