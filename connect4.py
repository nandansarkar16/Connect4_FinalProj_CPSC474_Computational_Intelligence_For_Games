import numpy as np, itertools

ROWS, COLS = 6, 7
ALL_MOVES = list(range(COLS))

class C4:
    # Immutable Connect-4 board - immutable  turn = 1 (X) or −1 (O)
    __slots__ = ("board", "turn")

    def __init__(self, board: np.ndarray | None = None, turn: int = 1):
        self.board = np.zeros((ROWS, COLS), np.int8) if board is None else board
        self.turn  = turn

    # mechanics
    def copy(self): return C4(self.board.copy(), self.turn)

    def legal(self): return [c for c in ALL_MOVES if self.board[0, c] == 0]

    def play(self, col: int):
        r = max(r for r in range(ROWS) if self.board[r, col] == 0)
        self.board[r, col] = self.turn
        self.turn *= -1

    # terminal test 
    def winner(self) -> int | None:
        B = self.board
        for r, c in itertools.product(range(ROWS), range(COLS)):
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                if 0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS:
                    if abs(sum(B[r + i*dr, c + i*dc] for i in range(4))) == 4:
                        return int(np.sign(sum(B[r + i*dr, c + i*dc] for i in range(4))))
        if (B != 0).all(): return 0  # draw
        return None                  # ongoing

    def __str__(self):
        sym = {0: ".", 1: "X", -1: "O"}
        rows = (" ".join(sym[x] for x in self.board[r]) for r in range(ROWS))
        return "\n".join(rows) + f"\nTurn: {'X' if self.turn == 1 else 'O'}"
