import numpy as np, torch
from connect4 import C4
from mcts import MCTS
from dqn import Net, encode

def mcts_player(sims, alpha):
    tree = MCTS(sims=sims, amaf_alpha=alpha)
    def play(state: C4):
        mv = tree.choose_move(state)
        tree.advance(mv)                  # shift root after we use the move
        return mv
    return play

def dqn_player1(path="dqn_final_50k.pt"):
    net = Net(); net.load_state_dict(torch.load(path)); net.eval()
    def play(s: C4):
        with torch.no_grad():
            q = net(torch.tensor(encode(s)[None]))
            legal = s.legal()
            return int(legal[np.argmax(q[0, legal])])
    return play

def dqn_player2(path="dqn_final_100k.pt"):
    net = Net(); net.load_state_dict(torch.load(path)); net.eval()
    def play(s: C4):
        with torch.no_grad():
            q = net(torch.tensor(encode(s)[None]))
            legal = s.legal()
            return int(legal[np.argmax(q[0, legal])])
    return play

def dqn_player3(path="dqn_final_200k.pt"):
    net = Net(); net.load_state_dict(torch.load(path)); net.eval()
    def play(s: C4):
        with torch.no_grad():
            q = net(torch.tensor(encode(s)[None]))
            legal = s.legal()
            return int(legal[np.argmax(q[0, legal])])
    return play

# battle tourney
def battle(p1, p2, n_games=50):
    w1 = w2 = d = 0
    for g in range(n_games):
        env = C4()
        first, second = (p1, p2) if g % 2 == 0 else (p2, p1)
        while True:
            agent = first if env.turn == 1 else second
            env.play(agent(env))
            res = env.winner()
            if res is None:
                continue
            if res == 1:
                if first is p1:
                    w1 += 1
                else:
                    w2 += 1
            elif res == -1:
                if second is p1:
                    w1 += 1
                else:
                    w2 += 1
            else:
                d += 1
            break
    return w1, w2, d

def main():
    agents = {
        "DQN1"         : dqn_player1(),
        "DQN2"         : dqn_player2(),
        "DQN3"         : dqn_player3(),
        "UCT"      : mcts_player(100000, 0.0),
        "AMAF80.5" : mcts_player(100000, 0.5),
        "AMAF81.0" : mcts_player(100000, 1.0),
    }
    names = list(agents)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            w1, w2, d = battle(agents[a], agents[b])
            print(f"{a:10s} vs {b:10s} : {w1}-{w2}-{d}")

if __name__ == "__main__":
    main()
