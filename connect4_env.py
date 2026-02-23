import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from typing import Optional


class EnvConnect4(gym.Env):

    def __init__(self):
        super().__init__()

        self.num_rows = 6
        self.num_cols = 7

        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.MultiDiscrete([3] * (self.num_rows * self.num_cols)),
                "turn": gym.spaces.Discrete(n=2, start=1),  # 1=X, 2=O
            }
        )

        self.pos_value_to_name = {0: "-", 1: "X", 2: "O"}
        self.col_id_to_name = {i: f"col-{i}" for i in range(self.num_cols)}

        self.action_space = gym.spaces.Discrete(self.num_cols)

        self.board = None
        self.turn = None
        self.count_moves = None

    def _get_obs(self):
        return {"board": self.board, "turn": self.turn}

    def _get_info(self):
        return {
            "board": self.board,
            "turn": self.turn,
            "legal columns": self._get_legal_actions(),
            "count moves": self.count_moves,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.board = [0] * (self.num_rows * self.num_cols)
        self.turn = 1
        self.count_moves = 0

        return self._get_obs(), self._get_info()

    def _idx(self, row, col):
        return row * self.num_cols + col

    def _get_drop_row(self, col):
        for row in range(self.num_rows - 1, -1, -1):
            if self.board[self._idx(row, col)] == 0:
                return row
        return None

    def _get_legal_actions(self):
        legal_cols = []
        for col in range(self.num_cols):
            if self.board[self._idx(0, col)] == 0:
                legal_cols.append(col)
        return legal_cols

    def is_winner(self, mark):
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

    def step(self, action):
        legal = self._get_legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action {action}. Legal columns are {legal}.")

        self.count_moves += 1

        row = self._get_drop_row(action)
        self.board[self._idx(row, action)] = self.turn

        reward = -0.01
        terminated = True if self.is_winner(mark=self.turn) or len(self._get_legal_actions()) == 0 else False
        truncated = False

        self.turn = 2 if self.turn == 1 else 1

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def print_current_board(self):
        print(f"Board after {self.count_moves} moves:")
        readable = [self.pos_value_to_name[v] for v in self.board]
        for r in range(self.num_rows):
            start = r * self.num_cols
            end = start + self.num_cols
            print(readable[start:end])
        print("[0, 1, 2, 3, 4, 5, 6] (column ids)")

    def check(self):
        try:
            check_env(self)
            print("Environment passes all checks!")
        except Exception as e:
            print(f"Environment has issues: {e}")