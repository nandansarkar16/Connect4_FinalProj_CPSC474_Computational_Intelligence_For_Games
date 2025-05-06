This is our CPSC 474: Computational Intelligence for Games Final Project on MCTS vs DQN for Connect 4
Authors: Nandan Sarkar and Andrew Pan



Environment setup and Test script:

Environment setup; Run these commands in the terminal to create a python virtual environment and install necessary packages:
python -m venv my_env
source my_env/bin/activate
pip install torch numpy tqdm

Test script:
python3 test_script.py 
(Note: So that this finishes in a few minutes, we had to change the number of simulations for each MCTS agent to 10k instead of 100k so results will differ)

Recreate full results:
To recreate full results with 100k simulations per move for each MCTS agent, run "python3 evaluate.py", however this takes a VERY long time, so we recommend either running it in the background using "nohup python3 evaluate.py > evaluation.txt 2>&1 &"
or running multiple parallel evaluation scripts evaluating only a couple agents at a time (you can do this by commenting out the agents not being evaluated in the main block of our evaluate.py file).



Description of Game: 
Connect 4 is a two-player, perfect information, deterministic, zero-sum game played on a 6-row by 7-column grid. 
Players alternate turns, dropping one of their colored discs into any of the seven columns. Each disc falls to the lowest available spot in its column. 
The goal is to be the first to form a line of four of one's own discs either horizontally, vertically, or diagonally.



Brief overview of code:
connect4.py: Implements the Connect‑4 board. Maintains a 6 × 7 NumPy array, generates legal moves, applies a move, and reports win, draw, or on going game.

mcts.py: Monte‑Carlo Tree Search player. Runs a fixed number of roll‑outs, stores visit counts and values, and re‑uses the tree/statistics from one step to the next via advance(). AMAF blending is controlled by one α parameter.

dqn.py: Defines a lightweight 3‑layer CNN plus a replay‑buffer DQN agent. Encoding turns the board into three feature planes — my stones, opponent stones, and side‑to‑move — so convolutions can detect winning patterns anywhere on the grid. 
Learns every step, uses Relu activations, and synchronises its target network every 100 updates for stable TD(0) training. 

train_dqn.py: Runs self-play training for DQN agent. The agent plays GAMES number of games against itself. Begins training after a 1000-step warmup and performs a learning update every 4 steps thereafter. Saves weights when finished.

evaluate.py: Plays a tournament between any mix of players (either different DQN agents or MCTS agents) and reports the wins/losses. MCTS agents use 100k simulations per move, and we play 50 games between agents.

show_game.py: Plays a single game between any two agents, printing the board after each move so you can inspect their behaviour turn‑by‑turn.

test_script.py: Same script as evaluate.py but a script for testing purposes for grading. To run in a few minutes, instead of many simulations, the MCTS uses only 10k simulations per move (as opposed to the 100k we use in evaluate), and we only play 10 games between agents (instead of 50 in evaluate).




Research Question: 



Results:
We evaluated 7 agents: DQN agents trained for 50k, 100k, 200k, and 500k games, as well as three MCTS variants — UCT (alpha=0), AMAF with alpha=0.5, and AMAF with alpha=1 all runing 100k simulations for each action. We played 50 games for each evaluation.

MCTS evaluations:
UCT vs AMAF0.5              : 25-25-0
UCT vs AMAF1.0              :
AMAF0.5 vs AMAF1.0          :


DQN vs MCTS evaluations:
DQN_50k_games vs UCT        : 25-25-0
DQN_50k_games vs AMAF0.5    : 25-25-0
DQN_50k_games vs AMAF1.0    : 14-36-0

DQN_100k_games vs UCT       : 25-25-0
DQN_100k_games vs AMAF0.5   : 25-25-0
DQN_100k_games vs AMAF1.0   : 6-44-0

DQN_200K_games vs UCT       :
DQN_200k_games vs AMAF0.5   :
DQN_200k_games vs AMAF1.0   :

DQN_500k_games vs UCT       : 25-25-0
DQN_500k_games vs AMAF0.5   : 25-25-0
DQN_500k_games vs AMAF1.0   : 23-27-0


DQN evaluations (deterministic: i.e. each game has the same play out): 
DQN1       vs DQN2          : 25-25-0 
DQN1       vs DQN3          : 25-25-0
DQN1       vs DQN4          : 25-25-0
DQN2       vs DQN3          : 25-25-0
DQN2       vs DQN4          : 25-25-0
DQN3       vs DQN4          : 25-25-0