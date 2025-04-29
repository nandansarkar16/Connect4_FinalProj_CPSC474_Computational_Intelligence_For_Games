import math, random, collections, numpy as np
from connect4 import C4, ALL_MOVES

# Hash helper: board bytes + 1-byte side-to-move
def key(state: C4) -> bytes:
    return state.board.tobytes() + bytes([1 if state.turn == 1 else 0])

class MCTS:
    """
    One object per game.  After every chosen move call
        mcts.advance(chosen_move)
    to shift the root to the corresponding child.
    """
    def __init__(self, sims=800, c=1.4, amaf_alpha=0.0):
        self.sims, self.c, self.alpha = sims, c, amaf_alpha

        # Global statistics tables  (persist across plies)
        self.Ns  = collections.defaultdict(int)
        self.Nsa = collections.defaultdict(int)
        self.Wsa = collections.defaultdict(float)

        self.N_amaf = collections.defaultdict(int)
        self.W_amaf = collections.defaultdict(float)

        self.root_key = None     # key of current root node

    def choose_move(self, root: C4):
        """
        Run self.sims simulations from `root`, return best action.
        Keeps all tree statistics for future moves.
        """
        self.root_key = key(root)                 # remember where root lives
        for _ in range(self.sims):
            self._simulate(root.copy())

        # pick action with highest visit-count
        counts = [self.Nsa[self.root_key, a] for a in ALL_MOVES]
        best = int(np.argmax(counts))
        return best

    def advance(self, played_move: int):
        """
        Shift the root to the child reached by `played_move`.
        Keeps all accumulated statistics.
        """
        if self.root_key is None:
            return                                      # first call not yet made

        # Reconstruct board corresponding to old root
        child_board = C4()
        child_board.board = (
            np.frombuffer(self.root_key[:-1], np.int8)  # read-only view
            .copy()                                     # make it writable
            .reshape(6, 7)
        )
        child_board.turn = 1 if self.root_key[-1] else -1
        child_board.play(played_move)                   # apply the move
        self.root_key = key(child_board)                # new root key

    # Internal: one simulation from current root
    def _simulate(self, node: C4):
        path, rollout_moves = [], []

        while True:
            k = key(node)
            legal = node.legal()

            # Expansion
            if self.Ns[k] == 0:
                a = random.choice(legal)
                path.append((k, a))
                node.play(a)
                break

            # Selection
            ucb_vals = []
            for a in legal:
                q = self.Wsa[k, a] / self.Nsa[k, a] if self.Nsa[k, a] else 0.0
                if self.alpha and self.N_amaf[k, a]:
                    q_amaf = self.W_amaf[k, a] / self.N_amaf[k, a]
                    q = (1 - self.alpha) * q + self.alpha * q_amaf
                u = self.c * math.sqrt(self.Ns[k]) / (1 + self.Nsa[k, a])
                ucb_vals.append(q + u)
            a = legal[int(np.argmax(ucb_vals))]
            path.append((k, a))
            node.play(a)

            if node.winner() is not None:
                break

        # Random rollout
        winner = node.winner()
        if winner is None:
            turn_at_leaf = node.turn
            while True:
                mv = random.choice(node.legal())
                node.play(mv)
                rollout_moves.append(mv)
                winner = node.winner()
                if winner is not None:
                    break
            winner *= turn_at_leaf
        value = winner                     # +1 / −1 / 0

        # Back prop
        seen_states = set()
        for k, a in path:
            if k not in seen_states:
                self.Ns[k] += 1
                seen_states.add(k)

            self.Nsa[k, a] += 1
            self.Wsa[k, a] += value

            for m in rollout_moves:
                self.N_amaf[k, m] += 1
                self.W_amaf[k, m] += value

            value = -value                # switch perspective
