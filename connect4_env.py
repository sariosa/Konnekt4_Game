# connect4_env.py

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from typing import Optional


class EnvConnect4(gym.Env):
    """
    Connect4 Gymnasium Env.

    Key design choice (IMPORTANT):
    - Rewards are from the CURRENT/ACTING player's perspective.
      * Win on your move: +1
      * Lose (only happens via illegal move here): -1
      * Draw: 0
      * Otherwise: 0

    This makes Q-learning consistent when the policy uses observation["turn"]
    and acts for whichever player's turn it is.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.num_rows = 6
        self.num_cols = 7

        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.MultiDiscrete([3] * (self.num_rows * self.num_cols)),
                "turn": gym.spaces.Discrete(n=2, start=1),  # 1 or 2
            }
        )

        self.action_space = gym.spaces.Discrete(self.num_cols)

        self.pos_value_to_name = {0: "-", 1: "X", 2: "O"}
        self.col_id_to_name = {i: f"col-{i}" for i in range(self.num_cols)}

        # Emoji rendering (console)
        self.player1_emoji = "🔵"  # X / Player 1
        self.player2_emoji = "🔴"  # O / Player 2 (human or AI)
        self.empty_emoji = "⚫"

        self.board = None
        self.turn = None
        self.count_moves = None

    # ---------- Core API helpers ----------

    def _get_obs(self):
        # Return a COPY so policies can't mutate env via observation
        return {"board": self.board.copy(), "turn": int(self.turn)}

    def _get_info(self, winner=0, is_draw=False):
        return {
            "board": self.board.copy(),
            "turn": int(self.turn),
            "legal_columns": self._get_legal_actions(),
            "count_moves": int(self.count_moves),
            "winner": int(winner),
            "is_draw": bool(is_draw),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.board = [0] * (self.num_rows * self.num_cols)
        self.turn = 1
        self.count_moves = 0

        return self._get_obs(), self._get_info()

    # ---------- Pure game logic ----------

    def _idx(self, row: int, col: int) -> int:
        return row * self.num_cols + col

    def _get_drop_row(self, col: int) -> int:
        # pieces fall to the bottom -> scan from bottom row up
        for row in range(self.num_rows - 1, -1, -1):
            if self.board[self._idx(row, col)] == 0:
                return row
        return -1

    def _get_legal_actions(self):
        # legal if top cell in the column is empty
        return [col for col in range(self.num_cols) if self.board[self._idx(0, col)] == 0]

    def is_winner(self, mark: int) -> bool:
        # Horizontal
        for row in range(self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row, col + i)] == mark for i in range(4)):
                    return True

        # Vertical
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols):
                if all(self.board[self._idx(row + i, col)] == mark for i in range(4)):
                    return True

        # Diagonal down-right
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row + i, col + i)] == mark for i in range(4)):
                    return True

        # Diagonal up-right
        for row in range(3, self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row - i, col + i)] == mark for i in range(4)):
                    return True

        return False

    # ---------- Gym step() ----------

    def step(self, action: int):
        legal = self._get_legal_actions()

        # Illegal move => acting player immediately loses
        if action not in legal:
            terminated = True
            truncated = False
            reward = -1.0  # acting player loses
            # winner is the OTHER player
            winner = 2 if self.turn == 1 else 1
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=winner, is_draw=False)

        self.count_moves += 1

        row = self._get_drop_row(action)
        if row == -1:
            # defensive fallback (should not happen if legal)
            terminated = True
            truncated = False
            reward = -1.0
            winner = 2 if self.turn == 1 else 1
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=winner, is_draw=False)

        # Apply move for current player
        self.board[self._idx(row, action)] = self.turn

        # Win check (current player just played)
        if self.is_winner(self.turn):
            terminated = True
            truncated = False
            reward = +1.0
            winner = self.turn
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=winner, is_draw=False)

        # Draw check
        if len(self._get_legal_actions()) == 0:
            terminated = True
            truncated = False
            reward = 0.0
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=0, is_draw=True)

        # Continue game
        self.turn = 2 if self.turn == 1 else 1
        terminated = False
        truncated = False
        reward = 0.0
        return self._get_obs(), reward, terminated, truncated, self._get_info(winner=0, is_draw=False)

    # ---------- Console helper ----------

    def print_current_board(self):
        """
        Prints a Connect-4 style board with emojis:
        empty = ⚫, player 1 (X) = 🔵, player 2 (O) = 🔴

        Note: Internally, row 0 is the TOP (because legality checks top cell),
        and pieces drop toward increasing row index. So for a natural display,
        we print from top row -> bottom row (0 -> num_rows-1).
        """
        print(f"\nBoard after {self.count_moves} moves:")

        # Column headers
        print("  ".join(map(str, range(self.num_cols))))
        print("-" * (self.num_cols * 3))

        for r in range(self.num_rows):
            row_cells = []
            for c in range(self.num_cols):
                v = self.board[self._idx(r, c)]
                if v == 1:
                    row_cells.append(self.player1_emoji)
                elif v == 2:
                    row_cells.append(self.player2_emoji)
                else:
                    row_cells.append(self.empty_emoji)
            print(" ".join(row_cells))

        print()

    def check(self):
        check_env(self)
