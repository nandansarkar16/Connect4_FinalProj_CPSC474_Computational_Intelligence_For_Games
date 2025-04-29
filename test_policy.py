from connect4 import C4
from mcts import MCTS
root = C4()
m = MCTS(sims=100)
pi = m.policy(root)
print(pi, pi.sum())  