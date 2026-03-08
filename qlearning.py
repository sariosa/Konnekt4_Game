# qlearning.py
# Tabular Q-Learning for Connect-4 (single file)
#
# Changes applied:
# 1) epsilon_min: 0.10 -> 0.02
# 2) epsilon_decay: 0.99999 -> 0.999995
# 3) Opponent during training: mixed always (70% heuristic, 30% random)
# 4) Reward shaping: threat penalty if agent leaves opponent an immediate winning move
#
# Usage:
#   python qlearning.py --train
#   python qlearning.py --train --episodes 200000
#   python qlearning.py --eval
#
# Importable:
#   from qlearning import train
#   agent = train(episodes=50000)

import argparse
import random
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import time


# ----------------------------
# Constants
# ----------------------------
ROWS = 6
COLS = 7

EMPTY = 0
OPP_PIECE = 1     # opponent
AGENT_PIECE = 2   # Q-learning agent

PLAYER_EMOJI = "🔵"
AI_EMOJI = "🔴"
EMPTY_EMOJI = "⚫"


# ----------------------------
# Utilities: board logic
# ----------------------------
def valid_actions(board: np.ndarray) -> List[int]:
    """Return columns that are not full."""
    return [c for c in range(COLS) if board[ROWS - 1, c] == EMPTY]


def next_open_row(board: np.ndarray, col: int) -> Optional[int]:
    """Return next available row in a column, or None if full."""
    for r in range(ROWS):
        if board[r, col] == EMPTY:
            return r
    return None


def apply_action(board: np.ndarray, col: int, piece: int) -> Optional[np.ndarray]:
    """Return new board after dropping piece into col; None if illegal."""
    if col < 0 or col >= COLS:
        return None
    if board[ROWS - 1, col] != EMPTY:
        return None

    r = next_open_row(board, col)
    if r is None:
        return None

    nb = board.copy()
    nb[r, col] = piece
    return nb


def winning_move(board: np.ndarray, piece: int) -> bool:
    """Check whether piece has four in a row."""
    # Horizontal
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r, c + i] == piece for i in range(4)):
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i, c] == piece for i in range(4)):
                return True

    # Diagonal down-right
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r + i, c + i] == piece for i in range(4)):
                return True

    # Diagonal up-right
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r - i, c + i] == piece for i in range(4)):
                return True

    return False


def is_draw(board: np.ndarray) -> bool:
    """Return True if there are no legal moves left."""
    return len(valid_actions(board)) == 0


def find_immediate_win_col(board: np.ndarray, piece: int) -> Optional[int]:
    """Return a column that wins immediately for piece, else None."""
    for a in valid_actions(board):
        nb = apply_action(board, a, piece)
        if nb is not None and winning_move(nb, piece):
            return a
    return None


def print_board(board: np.ndarray):
    """Pretty print board for console play."""
    flipped = np.flip(board, 0)
    print("  ".join(map(str, range(COLS))))
    print("-" * (COLS * 3))
    for r in range(ROWS):
        row_str = []
        for c in range(COLS):
            if flipped[r, c] == OPP_PIECE:
                row_str.append(PLAYER_EMOJI)
            elif flipped[r, c] == AGENT_PIECE:
                row_str.append(AI_EMOJI)
            else:
                row_str.append(EMPTY_EMOJI)
        print(" ".join(row_str))
    print()


# ----------------------------
# Opponents
# ----------------------------
def opponent_random(board: np.ndarray) -> Optional[int]:
    """Random legal move."""
    v = valid_actions(board)
    return random.choice(v) if v else None


def opponent_heuristic(board: np.ndarray, my_piece: int, opp_piece: int) -> Optional[int]:
    """
    Baseline heuristic:
    1) win now
    2) block opponent win
    3) prefer center
    4) random valid
    """
    v = valid_actions(board)
    if not v:
        return None

    w = find_immediate_win_col(board, my_piece)
    if w is not None:
        return w

    b = find_immediate_win_col(board, opp_piece)
    if b is not None:
        return b

    if 3 in v:
        return 3

    return random.choice(v)


# ----------------------------
# Q-Learning Agent (tabular)
# ----------------------------
@dataclass
class QParams:
    alpha: float = 0.10
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.999995
    epsilon_min: float = 0.02


class QLearningAgent:
    """Classic tabular Q-learning agent using the raw board as state."""

    def __init__(self, params: QParams):
        self.params = params
        self.epsilon = params.epsilon_start
        self.q: Dict[Tuple[bytes, int], np.ndarray] = defaultdict(
            lambda: np.zeros(COLS, dtype=np.float32)
        )

    def state_key(self, board: np.ndarray, turn: int) -> Tuple[bytes, int]:
        """State = full raw board + turn."""
        return (board.tobytes(), int(turn))

    def choose_action(self, board: np.ndarray, turn: int, legal: List[int]) -> Optional[int]:
        """Epsilon-greedy action selection with legal-action masking."""
        if not legal:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal)

        key = self.state_key(board, turn)
        qvals = self.q[key]
        return max(legal, key=lambda a: qvals[a])

    def update(
        self,
        s_board: np.ndarray,
        s_turn: int,
        action: int,
        reward: float,
        sp_board: np.ndarray,
        sp_turn: int,
        sp_legal: List[int],
        done: bool,
    ):
        """Standard Q-learning update."""
        key = self.state_key(s_board, s_turn)
        old = self.q[key][action]

        if done or not sp_legal:
            target = reward
        else:
            next_key = self.state_key(sp_board, sp_turn)
            target = reward + self.params.gamma * max(self.q[next_key][a] for a in sp_legal)

        self.q[key][action] = old + self.params.alpha * (target - old)

        if self.epsilon > self.params.epsilon_min:
            self.epsilon = max(
                self.params.epsilon_min,
                self.epsilon * self.params.epsilon_decay,
            )

    def save(self, filename: str = "q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q), f)
        print(f"Saved Q-table to {filename} (states={len(self.q)})")

    def load(self, filename: str = "q_table.pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.q = defaultdict(lambda: np.zeros(COLS, dtype=np.float32), data)
        print(f"Loaded Q-table from {filename} (states={len(self.q)})")


# ----------------------------
# Training
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def train(
    episodes: int = 200_000,
    seed: int = 42,
    print_every: int = 5000,
    q_file: str = "q_table.pkl",
    learning_curve_file: str = "learning_curve.png",
):
    """
    Train against a mixed opponent:
    70% heuristic, 30% random.
    """
    set_seed(seed)
    agent = QLearningAgent(QParams())

    wins = losses = draws = 0
    window = 1000
    recent = []
    curve = []

    print("Starting training...")

    for ep in range(1, episodes + 1):
        board = np.zeros((ROWS, COLS), dtype=np.int8)
        turn = 0  # 0 opponent, 1 agent
        done = False

        while not done:
            if not valid_actions(board):
                draws += 1
                done = True
                break

            if turn == 0:
                # Opponent step: mix always
                if random.random() < 0.70:
                    a = opponent_heuristic(board, my_piece=OPP_PIECE, opp_piece=AGENT_PIECE)
                else:
                    a = opponent_random(board)

                nb = apply_action(board, a, OPP_PIECE)
                if nb is None:
                    draws += 1
                    done = True
                    break

                board = nb

                if winning_move(board, OPP_PIECE):
                    losses += 1
                    done = True
                    break

                if is_draw(board):
                    draws += 1
                    done = True
                    break

                turn = 1

            else:
                # Agent step
                s_board = board.copy()
                legal = valid_actions(s_board)
                action = agent.choose_action(s_board, turn=1, legal=legal)

                nb = apply_action(board, action, AGENT_PIECE)
                if nb is None:
                    agent.update(
                        s_board, 1, action,
                        reward=-1.0,
                        sp_board=board, sp_turn=0,
                        sp_legal=valid_actions(board),
                        done=True,
                    )
                    losses += 1
                    done = True
                    break

                board = nb

                if winning_move(board, AGENT_PIECE):
                    wins += 1
                    agent.update(
                        s_board, 1, action,
                        reward=+1.0,
                        sp_board=board.copy(), sp_turn=0,
                        sp_legal=valid_actions(board),
                        done=True,
                    )
                    done = True
                    break

                if is_draw(board):
                    draws += 1
                    agent.update(
                        s_board, 1, action,
                        reward=0.0,
                        sp_board=board.copy(), sp_turn=0,
                        sp_legal=valid_actions(board),
                        done=True,
                    )
                    done = True
                    break

                # Reward shaping:
                # punish moves that leave an immediate opponent win
                threat_penalty = -0.5 if find_immediate_win_col(board, OPP_PIECE) is not None else 0.0

                agent.update(
                    s_board, 1, action,
                    reward=threat_penalty,
                    sp_board=board.copy(), sp_turn=0,
                    sp_legal=valid_actions(board),
                    done=False,
                )
                turn = 0

        # Moving window win-rate
        recent.append(1 if winning_move(board, AGENT_PIECE) else 0)
        if len(recent) > window:
            recent.pop(0)
        curve.append(sum(recent) / len(recent))

        if ep % print_every == 0:
            cum_win = 100.0 * wins / ep
            mov_win = 100.0 * curve[-1]
            print(
                f"Ep {ep}/{episodes} | CumWin% {cum_win:.2f} | "
                f"Win(last{window}) {mov_win:.2f}% | "
                f"W/L/D {wins}/{losses}/{draws} | eps {agent.epsilon:.4f}"
            )

    print("Training complete.")
    agent.save(q_file)

    plt.figure()
    plt.plot(curve)
    plt.xlabel("Episode")
    plt.ylabel(f"Win Rate (moving avg, window={window})")
    plt.title("Training Learning Curve (Tabular Q-learning)")
    plt.tight_layout()
    plt.savefig(learning_curve_file)
    print(f"Saved learning curve: {learning_curve_file}")
    # Auto-refresh evaluation plot after every training run
    evaluate(
        q_file=q_file,
        games=1000,
        seed=seed,
        out_plot="evaluation.png",
    )

    return agent


# ----------------------------
# Evaluation
# ----------------------------
def play_one_game(agent: QLearningAgent, opponent: str) -> str:
    """
    Returns:
    - 'W' = agent win
    - 'L' = agent loss
    - 'D' = draw
    """
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    turn = 0  # opponent starts

    old_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy evaluation

    while True:
        if not valid_actions(board):
            agent.epsilon = old_eps
            return "D"

        if turn == 0:
            if opponent == "random":
                a = opponent_random(board)
            elif opponent == "heuristic":
                a = opponent_heuristic(board, my_piece=OPP_PIECE, opp_piece=AGENT_PIECE)
            else:
                raise ValueError("opponent must be 'random' or 'heuristic'")

            board = apply_action(board, a, OPP_PIECE)

            if winning_move(board, OPP_PIECE):
                agent.epsilon = old_eps
                return "L"
            if is_draw(board):
                agent.epsilon = old_eps
                return "D"

            turn = 1

        else:
            a = agent.choose_action(board, turn=1, legal=valid_actions(board))
            board = apply_action(board, a, AGENT_PIECE)

            if winning_move(board, AGENT_PIECE):
                agent.epsilon = old_eps
                return "W"
            if is_draw(board):
                agent.epsilon = old_eps
                return "D"

            turn = 0


def evaluate(
    q_file: str = "q_table.pkl",
    games: int = 1000,
    seed: int = 42,
    out_plot: str = "evaluation.png",
):
    """Evaluate against random and heuristic opponents."""
    set_seed(seed)
    agent = QLearningAgent(QParams())
    agent.load(q_file)
    agent.epsilon = 0.0

    results = {}
    for opp in ["random", "heuristic"]:
        W = L = D = 0
        for _ in range(games):
            r = play_one_game(agent, opp)
            if r == "W":
                W += 1
            elif r == "L":
                L += 1
            else:
                D += 1
        results[opp] = {"W": W, "L": L, "D": D, "win_rate": W / games}

    print("\nEVALUATION (ε=0 greedy)")
    for opp, v in results.items():
        print(f"{opp}: W/L/D = {v['W']}/{v['L']}/{v['D']} | win_rate={v['win_rate']:.3f}")

    labels = ["Q vs Random", "Q vs Heuristic"]
    win_rates = [results["random"]["win_rate"], results["heuristic"]["win_rate"]]

    plt.figure()
    bars = plt.bar(labels, win_rates)
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    plt.title("Evaluation Win Rates (Tabular Q-learning)")
    # Add percentage labels above bars
    for bar, rate in zip(bars, win_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close
    print(f"Saved evaluation plot: {out_plot}")


# ----------------------------
# Optional: play vs human
# ----------------------------
def play_human_vs_ai(
    q_file: str = "q_table.pkl",
    thinking_delay: float = 0.3,
    tactical_override: bool = False,
):
    """
    tactical_override=False: pure learned policy
    tactical_override=True : demo mode (WIN > BLOCK > Q)
    """
    agent = QLearningAgent(QParams())
    agent.load(q_file)
    agent.epsilon = 0.0

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    turn = 0  # 0 human, 1 AI

    print_board(board)

    while True:
        if not valid_actions(board):
            print("IT'S A DRAW!")
            return

        if turn == 0:
            try:
                col = int(input(f"Your move {PLAYER_EMOJI} (0-{COLS-1}): ").strip())
            except ValueError:
                print("Enter a number 0-6.")
                continue

            if col not in range(COLS):
                print("Column out of range.")
                continue
            if col not in valid_actions(board):
                print("That column is full.")
                continue

            board = apply_action(board, col, OPP_PIECE)
            print_board(board)

            if winning_move(board, OPP_PIECE):
                print("YOU WIN!")
                return

            turn = 1

        else:
            print("AI is thinking...")
            time.sleep(thinking_delay)

            legal = valid_actions(board)

            if tactical_override:
                col = find_immediate_win_col(board, AGENT_PIECE)
                if col is None:
                    col = find_immediate_win_col(board, OPP_PIECE)
                if col is None:
                    col = agent.choose_action(board, 1, legal)
            else:
                col = agent.choose_action(board, 1, legal)

            board = apply_action(board, col, AGENT_PIECE)
            print_board(board)

            if winning_move(board, AGENT_PIECE):
                print("AI WINS!")
                return

            turn = 0


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=5000)
    parser.add_argument("--qfile", type=str, default="q_table.pkl")
    parser.add_argument("--learning_curve", type=str, default="learning_curve.png")
    parser.add_argument("--eval_plot", type=str, default="evaluation.png")
    parser.add_argument("--tactical_override", action="store_true")
    args = parser.parse_args()

    if not (args.train or args.eval or args.play):
        print("Choose one: --train and/or --eval and/or --play")
        return

    if args.train:
        train(
            episodes=args.episodes,
            seed=args.seed,
            print_every=args.print_every,
            q_file=args.qfile,
            learning_curve_file=args.learning_curve,
        )

    if args.eval:
        evaluate(
            q_file=args.qfile,
            games=args.games,
            seed=args.seed,
            out_plot=args.eval_plot,
        )

    if args.play:
        play_human_vs_ai(
            q_file=args.qfile,
            thinking_delay=0.3,
            tactical_override=args.tactical_override,
        )


if __name__ == "__main__":
    main()