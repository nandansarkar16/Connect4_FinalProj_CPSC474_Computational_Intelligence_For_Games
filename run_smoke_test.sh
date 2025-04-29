#!/usr/bin/env bash
python - <<'PY'
from connect4 import C4
from mcts import MCTS
root = C4()
pi = MCTS(sims=50).policy(root)
assert abs(pi.sum() - 1) < 1e-6
print("✅  MCTS smoke-test passed")
PY
