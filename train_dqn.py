import tqdm, torch
from connect4 import C4
from dqn import DQNAgent
import os

GAMES = 1000000         # total self-play games

def self_play_episode(agent: DQNAgent):
    env = C4()
    while True:
        a = agent.act(env)
        env_next = env.copy()
        env_next.play(a)
        w = env_next.winner()
        r = 0.0 if w is None else (1.0 if w == env.turn else -1.0)
        agent.remember(env, a, r, env_next, w is not None)
        if agent.steps > 1000 and agent.steps % 4 == 0: # train every 4 steps and after 1000 steps of warmup
            agent.learn()
        env = env_next
        if w is not None: break

def main():
    print(os.cpu_count())
    torch.set_num_threads(8)      # single Zoo node
    agent = DQNAgent()
    for _ in tqdm.trange(GAMES, desc="self-play"):
        self_play_episode(agent)
    torch.save(agent.policy.state_dict(), "dqn_final_1M.pt")

if __name__ == "__main__":
    main()