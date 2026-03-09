# connect4_env.py

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from typing import Optional


class EnvConnect4(gym.Env):
    """
    Gymnasium environment for the Connect 4 game.

    The environment uses a 6x7 board stored as a flattened list.
    Two players take turns dropping pieces into one of the columns.
    A move is valid if the selected column is not full.

    Reward design:
    - +1 for a winning move by the acting player
    - -1 for an illegal move by the acting player
    - 0 for a draw
    - 0 for all non-terminal intermediate moves

    The reward is always given from the perspective of the current
    acting player.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """
        Initializes the Connect 4 environment.

        Sets board dimensions, observation space, action space,
        display mappings, and internal game state variables.
        """
        super().__init__()

        self.num_rows = 6
        self.num_cols = 7

        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.MultiDiscrete([3] * (self.num_rows * self.num_cols)),
                "turn": gym.spaces.Discrete(n=2, start=1),
            }
        )

        self.action_space = gym.spaces.Discrete(self.num_cols)

        self.pos_value_to_name = {0: "-", 1: "X", 2: "O"}
        self.col_id_to_name = {i: f"col-{i}" for i in range(self.num_cols)}

        # Symbols used for console rendering
        self.player1_emoji = "🔵"
        self.player2_emoji = "🔴"
        self.empty_emoji = "⚫"

        self.board = None
        self.turn = None
        self.count_moves = None

    # ---------- Core API helpers ----------

    def _get_obs(self):
        """
        Returns the current observation.

        Returns
        -------
        dict
            Dictionary containing the current board state and the
            player whose turn it is.
        """
        # Return a copy to prevent external modification of the environment state
        return {"board": self.board.copy(), "turn": int(self.turn)}

    def _get_info(self, winner=0, is_draw=False):
        """
        Returns additional environment information.

        Parameters
        ----------
        winner : int, optional
            Winning player identifier. Default is 0 for no winner.
        is_draw : bool, optional
            Whether the current state is a draw. Default is False.

        Returns
        -------
        dict
            Additional information about the current game state.
        """
        return {
            "board": self.board.copy(),
            "turn": int(self.turn),
            "legal_columns": self._get_legal_actions(),
            "count_moves": int(self.count_moves),
            "winner": int(winner),
            "is_draw": bool(is_draw),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        tuple
            Observation and info dictionary for the initial state.
        """
        super().reset(seed=seed)

        self.board = [0] * (self.num_rows * self.num_cols)
        self.turn = 1
        self.count_moves = 0

        return self._get_obs(), self._get_info()

    # ---------- Pure game logic ----------

    def _idx(self, row: int, col: int) -> int:
        """
        Converts a two-dimensional board position into a flat index.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        int
            Index in the flattened board representation.
        """
        return row * self.num_cols + col

    def _get_drop_row(self, col: int) -> int:
        """
        Finds the row where a piece will land in a given column.

        Parameters
        ----------
        col : int
            Column index.

        Returns
        -------
        int
            Row index where the piece will be placed, or -1 if the
            column is full.
        """
        # Pieces fall to the bottom, so the scan starts from the last row
        for row in range(self.num_rows - 1, -1, -1):
            if self.board[self._idx(row, col)] == 0:
                return row
        return -1

    def _get_legal_actions(self):
        """
        Returns all legal actions from the current state.

        A column is legal if its top cell is still empty.

        Returns
        -------
        list[int]
            List of playable column indices.
        """
        return [col for col in range(self.num_cols) if self.board[self._idx(0, col)] == 0]

    def is_winner(self, mark: int) -> bool:
        """
        Checks whether a player has formed a connect-four.

        Parameters
        ----------
        mark : int
            Player identifier, usually 1 or 2.

        Returns
        -------
        bool
            True if the given player has four connected pieces,
            otherwise False.
        """
        # Horizontal check
        for row in range(self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row, col + i)] == mark for i in range(4)):
                    return True

        # Vertical check
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols):
                if all(self.board[self._idx(row + i, col)] == mark for i in range(4)):
                    return True

        # Diagonal down-right check
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row + i, col + i)] == mark for i in range(4)):
                    return True

        # Diagonal up-right check
        for row in range(3, self.num_rows):
            for col in range(self.num_cols - 3):
                if all(self.board[self._idx(row - i, col + i)] == mark for i in range(4)):
                    return True

        return False

    # ---------- Gym step() ----------

    def step(self, action: int):
        """
        Executes one move in the environment.

        Parameters
        ----------
        action : int
            Column selected by the acting player.

        Returns
        -------
        tuple
            Observation, reward, terminated flag, truncated flag,
            and info dictionary.
        """
        legal = self._get_legal_actions()

        # Illegal move: the acting player immediately loses
        if action not in legal:
            terminated = True
            truncated = False
            reward = -1.0
            winner = 2 if self.turn == 1 else 1
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=winner, is_draw=False)

        self.count_moves += 1

        row = self._get_drop_row(action)
        if row == -1:
            # Defensive fallback in case a full column is somehow selected
            terminated = True
            truncated = False
            reward = -1.0
            winner = 2 if self.turn == 1 else 1
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=winner, is_draw=False)

        # Apply the move for the current player
        self.board[self._idx(row, action)] = self.turn

        # Check whether the current player has won
        if self.is_winner(self.turn):
            terminated = True
            truncated = False
            reward = +1.0
            winner = self.turn
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=winner, is_draw=False)

        # If no legal actions remain, the game is a draw
        if len(self._get_legal_actions()) == 0:
            terminated = True
            truncated = False
            reward = 0.0
            return self._get_obs(), reward, terminated, truncated, self._get_info(winner=0, is_draw=True)

        # Otherwise, switch to the other player and continue
        self.turn = 2 if self.turn == 1 else 1
        terminated = False
        truncated = False
        reward = 0.0
        return self._get_obs(), reward, terminated, truncated, self._get_info(winner=0, is_draw=False)

    # ---------- Console helper ----------

    def print_current_board(self):
        """
        Prints the current board state in the console.

        Display symbols:
        - empty cell: ⚫
        - player 1: 🔵
        - player 2: 🔴

        The board is printed from top row to bottom row for a natural
        Connect 4 display.
        """
        print(f"\nBoard after {self.count_moves} moves:")

        # Print column indices for user reference
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
        """
        Runs Gymnasium's environment checker on the environment.
        """
        check_env(self)
