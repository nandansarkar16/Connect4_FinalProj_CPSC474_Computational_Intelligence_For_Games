import numpy as np, torch
from connect4 import C4
from mcts import MCTS            
from dqn import Net, encode

def mcts(sims, alpha):
    tree = MCTS(sims=sims, amaf_alpha=alpha)
    def play(state: C4):
        move = tree.choose_move(state)   
        tree.advance(move)           
        return move
    return play

def dqn_player_1(path="./weights/dqn_final_100k.pt"):
    net = Net(); net.load_state_dict(torch.load(path)); net.eval()
    def play(s: C4):
        with torch.no_grad():
            q = net(torch.tensor(encode(s)[None]))
            legal = s.legal()
            return int(legal[np.argmax(q[0, legal])])
    return play

def dqn_player_2(path="dqn_final_100k_v2.pt"):
    net = Net(); net.load_state_dict(torch.load(path)); net.eval()
    def play(s: C4):
        with torch.no_grad():
            q = net(torch.tensor(encode(s)[None]))
            legal = s.legal()
            return int(legal[np.argmax(q[0, legal])])
    return play

env = C4()
p1  = dqn_player_1()
p2  = dqn_player_2()

while env.winner() is None:
    mv = p1(env) if env.turn == 1 else p2(env)
    env.play(mv)
    print(env, '\n')
