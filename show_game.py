import numpy as np, torch
from connect4 import C4
from mcts import MCTS            
from dqn import Net, encode

def persistent_mcts(sims, alpha):
    tree = MCTS(sims=sims, amaf_alpha=alpha)

    def play(state: C4):
        move = tree.choose_move(state)   # search from current root
        tree.advance(move)               # shift root after we use the move
        return move

    return play

def dqn_player(path="dqn_final_200k.pt"):
    net = Net(); net.load_state_dict(torch.load(path)); net.eval()
    def play(s: C4):
        with torch.no_grad():
            q = net(torch.tensor(encode(s)[None]))
            legal = s.legal()
            return int(legal[np.argmax(q[0, legal])])
    return play

env = C4()
p2  = persistent_mcts(80000, 1.0)   # MCTS with 20k simulations and AMAF alpha=1.0
p1  = dqn_player()                   # DQN as before, 1000 games trained

while env.winner() is None:
    mv = p1(env) if env.turn == 1 else p2(env)
    env.play(mv)
    print(env, '\n')
