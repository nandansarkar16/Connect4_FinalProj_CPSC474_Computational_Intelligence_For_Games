This is our CPSC 474: Computational Intelligence for Games Final Project on MCTS vs DQN for Connect 4
Authors: Nandan Sarkar (ns956) and Andrew Pan (ap2722)



Environment setup and Test script:

Environment setup; Run these commands in the terminal to create a python virtual environment and install necessary packages:
python -m venv my_env
source my_env/bin/activate
pip install torch numpy tqdm

Test script:
python3 test_script.py 
(Note: So that this finishes in a few minutes, we had to change the number of simulations for each MCTS agent to 1k instead of 100k so results will differ from our main results)

Recreate full results:
To recreate full results with 100k simulations per move for each MCTS agent, run "python3 evaluate.py", however this takes a VERY long time, so we recommend either running it in the background using "nohup python3 evaluate.py > evaluation.txt 2>&1 &"
or running multiple parallel evaluation scripts evaluating only a couple agents at a time (you can do this by commenting out the agents not being evaluated in the main block of our evaluate.py file).



Description of Game: 
Connect 4 is a two-player, perfect information, deterministic, zero-sum game played on a 6-row by 7-column grid. 
Players alternate turns, dropping one of their colored discs into any of the seven columns. Each disc falls to the lowest available spot in its column. 
The goal is to be the first to form a line of four of one's own discs either horizontally, vertically, or diagonally.



Research Question: We're researching how DQN compares to MCTS on Connect 4.



Brief overview of code:
connect4.py: Implements the Connect‑4 board. Maintains a 6 × 7 NumPy array, generates legal moves, applies a move, and reports win, draw, or on going game.

mcts.py: Monte‑Carlo Tree Search player. Runs a fixed number of roll‑outs, stores visit counts and values, and re‑uses the tree/statistics from one step to the next via advance(). AMAF blending is controlled by one α parameter.

dqn.py: Defines a lightweight 3‑layer CNN plus a replay‑buffer DQN agent. Encoding turns the board into three feature planes — my stones, opponent stones, and side‑to‑move — so convolutions can detect winning patterns anywhere on the grid. 
Learns every step, uses Relu activations, and synchronises its target network every 100 updates for stable TD(0) training. 

train_dqn.py: Runs self-play training for DQN agent. The agent plays GAMES number of games against itself. Begins training after a 1000-step warmup and performs a learning update every 4 steps thereafter. Saves weights when finished.

evaluate.py: Plays a tournament between any mix of players (either different DQN agents or MCTS agents) and reports the wins/losses. MCTS agents use 100k simulations per move, and we play 50 games between agents. We switch which agent starts playing first each game.

show_game.py: Plays a single game between any two agents, printing the board after each move so you can inspect their behaviour turn‑by‑turn.

test_script.py: Same script as evaluate.py but a script for testing purposes for grading. To run in a few minutes, instead of many simulations, the MCTS uses only 1k simulations per move (as opposed to the 100k we use in evaluate), and we only play 10 games between agents (instead of 50 in evaluate).



Results:
We evaluated 7 agents: DQN agents trained for 50k, 100k, 200k, and 500k games, as well as three MCTS variants — UCT (alpha=0), AMAF with alpha=0.5, and AMAF with alpha=1 all runing 100k simulations for each action. We played 50 games for each evaluation.

MCTS evaluations:
UCT vs AMAF0.5              : 25-25-0
UCT vs AMAF1.0              : 25-25-0
AMAF0.5 vs AMAF1.0          : 19-31-0


DQN vs MCTS evaluations:
DQN_50k_games vs UCT        : 25-25-0
DQN_50k_games vs AMAF0.5    : 25-25-0
DQN_50k_games vs AMAF1.0    : 14-36-0

DQN_100k_games vs UCT       : 25-25-0
DQN_100k_games vs AMAF0.5   : 25-25-0
DQN_100k_games vs AMAF1.0   : 6-44-0

DQN_200K_games vs UCT       : 25-25-0
DQN_200k_games vs AMAF0.5   : 25-25-0
DQN_200k_games vs AMAF1.0   : 17-33-0

DQN_500k_games vs UCT       : 25-25-0
DQN_500k_games vs AMAF0.5   : 25-25-0
DQN_500k_games vs AMAF1.0   : 23-27-0


DQN evaluations: 
DQN1       vs DQN2          : 25-25-0 
DQN1       vs DQN3          : 25-25-0
DQN1       vs DQN4          : 25-25-0
DQN2       vs DQN3          : 25-25-0
DQN2       vs DQN4          : 25-25-0
DQN3       vs DQN4          : 25-25-0



Discussion: 
One of the most consistent patterns we observed was the strong advantage of going first. In nearly every matchup between agents of similar strength, the first player tended to win the game. This explains the frequent 25–25 scorelines we saw across many evaluations.
Using our show_game.py script to visualize individual games, we noticed that DQN agents often failed to block threats or capitalize on opportunities, such as completing three-in-a-row setups. These mistakes were especially common for agents trained with fewer games. However, the quality of play improved steadily as we increased the training set size, with the 500k-game DQN performing noticeably better than the 50k or 100k versions. This trend aligns with our expectations: more training leads to better generalization and strategic play. With additional computational resources, we expect that training DQN agents on even more games would yield significantly stronger performance.
Among all agents evaluated, AMAF with alpha = 1.0 consistently emerged as the strongest, both quantitatively (in terms of win/loss records) and qualitatively (based on observed gameplay). This result is intuitive: by fully incorporating information from all rollouts, AMAF1.0 builds a more informed and robust search tree than UCT or partial AMAF blends. For example, it defeated the 100k DQN by a wide margin (44–6), and the 50k DQN by 36–14.
In terms of runtime, the difference between the two approaches was stark. DQN agents, once trained, were extremely fast at inference, requiring only a forward pass through a small CNN to choose actions. MCTS agents, on the other hand, were much slower. With 100k simulations per move, each game took several minutes to play out, and the full evaluation tournament spanned multiple days. This highlights a key tradeoff we discussed in lecture: DQN demands heavy up-front training time but enables fast, scalable inference, whereas MCTS is slower per move but requires no training phase.
In summary, given our current computational constraints, AMAF with alpha = 1.0 was the best-performing agent across our experiments. Furthermore, we believe that with extended training (e.g., into the millions of games), DQN agents would surpass MCTS in performance (and of course inference efficiency as discussed earlier) by leveraging learned generalization rather than time consuming brute-force simulation.