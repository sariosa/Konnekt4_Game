# qlearning.py
# Feature-based Q-Learning for Connect-4
#
# Usage:
#   python qlearning.py --train
#   python qlearning.py --train --episodes 300000
#   python qlearning.py --eval
#
# Importable:
#   from qlearning import train
#   agent = train(episodes=300000)

import argparse
import random
import pickle
from typing import List
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from connect4_env import EnvConnect4
from policies import PolicyQLearning, PolicyHeuristic, PolicyRandom


# ----------------------------
# Constants
# ----------------------------
ROWS = 6
COLS = 7


# ----------------------------
# Feature-based state utilities
# ----------------------------
def _canonical_board(board: list) -> tuple:
    """
    Return a left-right canonical board form.
    This lets mirrored positions share one representation.
    """
    arr = np.array(board).reshape(ROWS, COLS)
    mirror = np.fliplr(arr).flatten().tolist()
    return tuple(board) if board <= mirror else tuple(mirror)


def _col_height(board: list, col: int) -> int:
    """Return the number of occupied cells in a column."""
    return sum(1 for r in range(ROWS) if board[r * COLS + col] != 0)


def _count_n_in_a_row(board: list, piece: int, n: int) -> int:
    """
    Count windows of exactly n pieces and (4-n) empty cells.
    Used to detect 2-in-a-row and 3-in-a-row patterns.
    """
    arr = np.array(board).reshape(ROWS, COLS)
    empty = 0
    count = 0

    def chk(w):
        return w.count(piece) == n and w.count(empty) == (4 - n)

    # Horizontal
    for c in range(COLS - 3):
        for r in range(ROWS):
            if chk([arr[r, c + i] for i in range(4)]):
                count += 1

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if chk([arr[r + i, c] for i in range(4)]):
                count += 1

    # Diagonal down-right
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if chk([arr[r + i, c + i] for i in range(4)]):
                count += 1

    # Diagonal up-right
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if chk([arr[r - i, c + i] for i in range(4)]):
                count += 1

    return count


def _immediate_win_cols(env: EnvConnect4, piece: int) -> List[int]:
    """
    Return legal columns that give an immediate win for the given piece.
    """
    wins = []
    for col in env._get_legal_actions():
        row = env._get_drop_row(col)
        if row == -1:
            continue
        idx = env._idx(row, col)
        env.board[idx] = piece
        if env.is_winner(piece):
            wins.append(col)
        env.board[idx] = 0
    return wins


def board_to_features(board: list, turn: int, env: EnvConnect4) -> tuple:
    """
    Convert a raw board into a compact strategic feature representation.

    Features include:
    - current turn
    - coarse column heights
    - own and opponent 2/3-in-a-row patterns
    - immediate winning threats
    - center control
    - per-column win flags
    - coarse game phase
    """
    cb = list(_canonical_board(board))
    piece = turn
    opp = 2 if turn == 1 else 1

    feats = [int(turn)]

    # Column heights (bucketed)
    for c in range(COLS):
        feats.append(min(_col_height(cb, c) // 2, 3))

    # Own patterns
    feats.append(min(_count_n_in_a_row(cb, piece, 2), 8))
    feats.append(min(_count_n_in_a_row(cb, piece, 3), 4))

    # Immediate winning moves
    orig_board = env.board.copy() if env.board is not None else [0] * (ROWS * COLS)
    try:
        env.board = cb.copy()
        agent_threats = _immediate_win_cols(env, piece)
        opp_threats = _immediate_win_cols(env, opp)
    finally:
        env.board = orig_board

    feats.append(min(len(agent_threats), 3))

    # Opponent patterns
    feats.append(min(_count_n_in_a_row(cb, opp, 2), 8))
    feats.append(min(_count_n_in_a_row(cb, opp, 3), 4))
    feats.append(min(len(opp_threats), 3))

    # Center control
    center = [cb[r * COLS + 3] for r in range(ROWS) if cb[r * COLS + 3] != 0]
    if not center:
        feats.append(0)
    elif all(p == piece for p in center):
        feats.append(1)
    elif all(p == opp for p in center):
        feats.append(2)
    else:
        feats.append(3)

    # Per-column immediate win flags
    for c in range(COLS):
        feats.append(1 if c in agent_threats else 0)
    for c in range(COLS):
        feats.append(1 if c in opp_threats else 0)

    # Game phase
    total = sum(1 for x in cb if x != 0)
    feats.append(min(total // 4, 10))

    return tuple(feats)


# ----------------------------
# Q-Learning policy wrapper
# ----------------------------
class PolicyQLearningV4(PolicyQLearning):
    """
    Q-learning policy using the feature-based state representation
    instead of the full raw board.
    """
    def __init__(self, env: EnvConnect4, alpha=0.1, gamma=0.99, epsilon=1.0, seed: int = 42):
        super().__init__(env, alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
        self._env_ref = env

    def _state_key(self, observation):
        board = list(observation["board"])
        turn = int(observation["turn"])
        return board_to_features(board, turn, self._env_ref)

    def _get_action(self, env, observation):
        # Keep the feature extractor tied to the current environment
        self._env_ref = env

        s = self._state_key(observation)
        self._ensure(s)

        legal = env._get_legal_actions()

        # Exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))

        # Exploitation with legal-action masking
        q = self.Q[s].copy()
        mask = np.full(self.n_actions, -np.inf, dtype=np.float32)
        mask[legal] = q[legal]
        return int(np.argmax(mask))

    def save(self, filename: str = "q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)
        print(f"Saved Q-table → {filename}  (states={len(self.Q)})")

    def load(self, filename: str = "q_table.pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.Q = data
        print(f"Loaded Q-table ← {filename}  (states={len(self.Q)})")


# ----------------------------
# Reward-shaping helper
# ----------------------------
def _has_safe_move(env: EnvConnect4, acting_piece: int, opp_piece: int) -> bool:
    """
    Return True if the acting player has at least one move
    that does not leave an immediate winning reply.
    """
    for col in env._get_legal_actions():
        row = env._get_drop_row(col)
        if row == -1:
            continue
        idx = env._idx(row, col)
        env.board[idx] = acting_piece
        opp_wins = _immediate_win_cols(env, opp_piece)
        env.board[idx] = 0
        if not opp_wins:
            return True
    return False


# ----------------------------
# Training configuration
# ----------------------------
@dataclass
class TrainParams:
    episodes: int = 200_000
    alpha_start: float = 0.15
    alpha_end: float = 0.01
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.999996
    epsilon_min: float = 0.05
    heuristic_prob: float = 0.60
    seed: int = 42
    print_every: int = 5_000
    q_file: str = "q_table.pkl"
    curve_file: str = "learning_curve.png"


# ----------------------------
# Training
# ----------------------------
def train(
    episodes: int = 200_000,
    seed: int = 42,
    print_every: int = 5_000,
    q_file: str = "q_table.pkl",
    curve_file: str = "learning_curve.png",
) -> PolicyQLearningV4:
    """
    Train and return a feature-based Q-learning agent.
    """
    params = TrainParams(
        episodes=episodes,
        seed=seed,
        print_every=print_every,
        q_file=q_file,
        curve_file=curve_file,
    )

    random.seed(seed)
    np.random.seed(seed)

    env = EnvConnect4()
    heuristic = PolicyHeuristic(seed=seed)
    rand_pol = PolicyRandom()

    agent = PolicyQLearningV4(
        env,
        alpha=params.alpha_start,
        gamma=params.gamma,
        epsilon=params.epsilon_start,
        seed=seed,
    )

    wins = losses = draws = 0
    window = 1000
    recent: List[int] = []
    curve: List[float] = []

    print(
        f"Training v4 | episodes={episodes} | ε_decay={params.epsilon_decay} "
        f"| α {params.alpha_start}→{params.alpha_end} "
        f"| heuristic_prob={params.heuristic_prob}"
    )

    for ep in range(1, episodes + 1):
        # Linear alpha decay
        frac = ep / episodes
        agent.alpha = params.alpha_start + frac * (params.alpha_end - params.alpha_start)

        obs, info = env.reset()
        episode_over = False
        last_obs = None
        last_action = None
        info_next = info

        while not episode_over:
            current_turn = env.turn
            legal = env._get_legal_actions()

            if not legal:
                draws += 1
                break

            # Alternate the side controlled by the agent
            agent_turn = (current_turn == 1 and ep % 2 != 0) or (current_turn == 2 and ep % 2 == 0)

            if agent_turn:
                s = agent._state_key(obs)
                action = agent._get_action(env, obs)

                last_obs = obs
                last_action = action

                piece = current_turn
                opp_piece = 2 if piece == 1 else 1
                must_block = _immediate_win_cols(env, opp_piece)

                obs_next, reward, terminated, truncated, info_next = env.step(action)
                episode_over = terminated or truncated

                s_next = agent._state_key(obs_next)
                legal_next = info_next["legal_columns"]

                if terminated:
                    winner = info_next["winner"]
                    if winner == piece:
                        shaped_reward = +5.0
                        wins += 1
                    elif winner != 0:
                        shaped_reward = -5.0
                        losses += 1
                    else:
                        shaped_reward = 0.0
                        draws += 1
                else:
                    # Reward correct blocking and punish leaving obvious threats
                    block_bonus = +2.0 if (must_block and action in must_block) else 0.0

                    opp_threats_after = _immediate_win_cols(env, opp_piece)
                    if opp_threats_after:
                        threat_penalty = -3.0 if _has_safe_move(env, piece, opp_piece) else 0.0
                    else:
                        threat_penalty = 0.0

                    own_threats = _immediate_win_cols(env, piece)
                    threat_bonus = 0.2 * min(len(own_threats), 3)

                    shaped_reward = -0.01 + block_bonus + threat_penalty + threat_bonus

                agent.update(s, action, shaped_reward, s_next, legal_next, episode_over)

                if agent.epsilon > params.epsilon_min:
                    agent.epsilon = max(params.epsilon_min, agent.epsilon * params.epsilon_decay)

                obs = obs_next

            else:
                # Opponent turn: mixed random / heuristic training
                if random.random() < params.heuristic_prob:
                    action = heuristic._get_action(env, obs)
                else:
                    action = rand_pol._get_action(env, obs)

                obs_next, reward, terminated, truncated, info_next = env.step(action)
                episode_over = terminated or truncated

                # If the opponent wins, penalize the agent's last move
                if terminated and info_next["winner"] == current_turn:
                    losses += 1
                    if last_obs is not None and last_action is not None:
                        s_last = agent._state_key(last_obs)
                        s_next_last = agent._state_key(obs_next)
                        agent.update(s_last, last_action, -5.0, s_next_last, [], True)
                elif terminated and info_next["is_draw"]:
                    draws += 1

                obs = obs_next

        # Track moving win rate
        recent.append(1 if info_next.get("winner", 0) in ([1] if ep % 2 != 0 else [2]) else 0)
        if len(recent) > window:
            recent.pop(0)
        curve.append(sum(recent) / len(recent))

        if ep % print_every == 0:
            cum_win = 100.0 * wins / ep
            mov_win = 100.0 * curve[-1]
            print(
                f"Ep {ep:>7}/{episodes} | "
                f"CumWin%={cum_win:5.2f} | "
                f"Win(last{window})={mov_win:5.2f}% | "
                f"W/L/D {wins}/{losses}/{draws} | "
                f"ε={agent.epsilon:.4f}  α={agent.alpha:.4f} | "
                f"states={len(agent.Q)}"
            )

    print("\nTraining complete.")
    agent.save(q_file)

    # Learning curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(curve, linewidth=0.6, alpha=0.4, color="steelblue", label="Win rate (raw)")
    if len(curve) > 5000:
        n = 5000
        smoothed = np.convolve(curve, np.ones(n) / n, mode="valid")
        ax.plot(
            range(n // 2, n // 2 + len(smoothed)),
            smoothed,
            linewidth=2.5,
            color="coral",
            label=f"Smoothed (n={n})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Win Rate (moving avg, window={window})")
    ax.set_title("Training Learning Curve — Q-learning v4 (Feature-Based States)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(curve_file, dpi=150)
    print(f"Saved learning curve → {curve_file}")

    return agent


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(q_file: str = "q_table.pkl", games: int = 1000, seed: int = 42):
    """
    Evaluate the trained agent greedily against random and heuristic opponents.
    """
    env = EnvConnect4()
    agent = PolicyQLearningV4(env, seed=seed)
    agent.load(q_file)
    agent.epsilon = 0.0

    heuristic = PolicyHeuristic(seed=seed)
    rand_pol = PolicyRandom()

    results = {}
    for opp_name, opp_pol in [("random", rand_pol), ("heuristic", heuristic)]:
        W = L = D = 0
        for i in range(games):
            obs, info = env.reset()
            done = False
            agent_is_p1 = (i % 2 == 0)

            while not done:
                is_agent = (env.turn == 1 and agent_is_p1) or (env.turn == 2 and not agent_is_p1)

                if is_agent:
                    action = agent._get_action(env, obs)
                else:
                    action = opp_pol._get_action(env, obs)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            agent_piece = 1 if agent_is_p1 else 2
            winner = info["winner"]
            if winner == agent_piece:
                W += 1
            elif info["is_draw"]:
                D += 1
            else:
                L += 1

        results[opp_name] = {"W": W, "L": L, "D": D, "win_rate": W / games}

    print("\nEVALUATION (ε=0 greedy)")
    for opp, v in results.items():
        print(
            f"  vs {opp:>10}: W/L/D = {v['W']:>4}/{v['L']:>4}/{v['D']:>4} "
            f"| win_rate = {v['win_rate']:.3f}"
        )

    labels = ["Q vs Random", "Q vs Heuristic"]
    win_rates = [results["random"]["win_rate"], results["heuristic"]["win_rate"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, win_rates, color=["steelblue", "coral"], width=0.5)
    for bar, wr in zip(bars, win_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            wr + 0.012,
            f"{wr:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
        )
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Evaluation Win Rates — Q-learning v4")
    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=150)
    print("Saved evaluation plot → evaluation.png")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Q-learning v4 for Connect-4")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=5_000)
    parser.add_argument("--qfile", type=str, default="q_table.pkl")
    parser.add_argument("--curve", type=str, default="learning_curve.png")
    args = parser.parse_args()

    if not (args.train or args.eval):
        print("Specify --train and/or --eval")
        return

    if args.train:
        train(
            episodes=args.episodes,
            seed=args.seed,
            print_every=args.print_every,
            q_file=args.qfile,
            curve_file=args.curve,
        )

    if args.eval:
        evaluate(q_file=args.qfile, games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()