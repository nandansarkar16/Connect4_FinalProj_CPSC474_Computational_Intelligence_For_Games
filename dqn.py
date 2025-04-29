import math, random, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from connect4 import C4, ROWS, COLS

# -------- state → tensor --------
def encode(state: C4) -> np.ndarray:
    p = np.zeros((3, ROWS, COLS), np.float32)
    p[0][state.board ==  state.turn]  = 1.0
    p[1][state.board == -state.turn] = 1.0
    p[2].fill(1.0 if state.turn == 1 else 0.0)
    return p

# CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.head = nn.Linear(64 * ROWS * COLS, COLS)

    def forward(self, x):                    # (B,3,6,7)
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return self.head(x.flatten(1))       # (B,7)

# Replay buffer
class Replay:
    def __init__(self, cap=200_000):
        self.buf = collections.deque(maxlen=cap)
    def push(self, *exp): self.buf.append(exp)
    def sample(self, n):  return random.sample(self.buf, n)
    def __len__(self):    return len(self.buf)

# agent
class DQNAgent:
    def __init__(self, lr=1e-3, gamma=.99,
                 eps_start=1.0, eps_end=.1, eps_decay=5_000):
        self.policy = Net()
        self.target = Net(); self.target.load_state_dict(self.policy.state_dict())
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma, self.replay = gamma, Replay()
        self.steps = 0
        self.eps_s, self.eps_e, self.decay = eps_start, eps_end, eps_decay

    # acting
    def epsilon(self) -> float:
        return self.eps_e + (self.eps_s - self.eps_e) * math.exp(-self.steps / self.decay)

    def act(self, s: C4) -> int:
        self.steps += 1
        if random.random() < self.epsilon():
            return random.choice(s.legal())
        with torch.no_grad():
            q = self.policy(torch.tensor(encode(s)[None]))
            legal = s.legal()
            return int(legal[torch.argmax(q[0, legal])])

    # storage
    def remember(self, s, a, r, s2, done):
        self.replay.push(encode(s), a, r, encode(s2), done)

    # learning
    def learn(self, batch=128):
        if len(self.replay) < batch: return
        S, A, R, S2, D = zip(*self.replay.sample(batch))
        S  = torch.tensor(S)
        S2 = torch.tensor(S2)
        A  = torch.tensor(A,  dtype=torch.int64)
        R  = torch.tensor(R,  dtype=torch.float32)
        D  = torch.tensor(D,  dtype=torch.float32)

        q_sa = self.policy(S).gather(1, A[:, None]).squeeze()
        with torch.no_grad():
            q_max = self.target(S2).max(1)[0]
            tgt = R + self.gamma * q_max * (1 - D)
        loss = F.mse_loss(q_sa, tgt)

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if self.steps % 500 == 0:
            self.target.load_state_dict(self.policy.state_dict())
