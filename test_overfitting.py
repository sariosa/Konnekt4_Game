# test_overfitting.py
# Overfitting / generalization test for the Connect-4 Q-learning agent.
#
# This script:
# - loads a trained Q-table
# - evaluates against training-like opponents
# - evaluates against held-out opponents
# - compares average win rates
# - flags large opponent-specific weaknesses
#
# Usage:
#   python test_overfitting.py --qfile q_table.pkl --games 300 --seeds 5

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from connect4_env import EnvConnect4
from policies import PolicyRandom, PolicyHeuristic
from qlearning import PolicyQLearningV4


# ----------------------------
# Extra held-out opponents
# ----------------------------
class PolicyCenterFirst:
    """
    Opponent that prefers the center, then near-center columns.
    """
    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def _get_action(self, env, observation=None):
        legal = env._get_legal_actions()
        center = env.num_cols // 2
        scores = [-abs(c - center) for c in legal]
        best = max(scores)
        best_moves = [c for c, s in zip(legal, scores) if s == best]
        return int(self.rng.choice(best_moves))


class PolicyBlockThenCenter:
    """
    Opponent that first blocks immediate losses,
    then prefers the center.
    """
    def __init__(self, seed: int = 456):
        self.rng = np.random.default_rng(seed)

    def _get_action(self, env, observation):
        legal = env._get_legal_actions()
        player = observation["turn"]
        opponent = 2 if player == 1 else 1

        # First block opponent immediate wins
        blocking_moves = []
        for col in legal:
            row = env._get_drop_row(col)
            if row == -1:
                continue
            idx = env._idx(row, col)

            env.board[idx] = opponent
            if env.is_winner(opponent):
                blocking_moves.append(col)
            env.board[idx] = 0

        if blocking_moves:
            return int(self.rng.choice(blocking_moves))

        # Otherwise prefer center
        center = env.num_cols // 2
        scores = [-abs(c - center) for c in legal]
        best = max(scores)
        best_moves = [c for c, s in zip(legal, scores) if s == best]
        return int(self.rng.choice(best_moves))


# ----------------------------
# Evaluation helpers
# ----------------------------
@dataclass
class EvalResult:
    wins: int
    losses: int
    draws: int
    win_rate: float
    draw_rate: float
    loss_rate: float


def play_one_game(env, agent, opponent, agent_is_p1: bool):
    """
    Play one game between the trained agent and one opponent.
    """
    agent._env_ref = env

    obs, info = env.reset()
    done = False

    while not done:
        is_agent_turn = (env.turn == 1 and agent_is_p1) or (env.turn == 2 and not agent_is_p1)

        if is_agent_turn:
            action = agent._get_action(env, obs)
        else:
            action = opponent._get_action(env, obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    agent_piece = 1 if agent_is_p1 else 2
    winner = info["winner"]

    if winner == agent_piece:
        return "W"
    elif info["is_draw"]:
        return "D"
    else:
        return "L"


def evaluate_matchup(agent, opponent_factory, games: int, seed: int) -> EvalResult:
    """
    Evaluate the agent against one opponent type over multiple games.
    """
    random.seed(seed)
    np.random.seed(seed)

    env = EnvConnect4()
    agent._env_ref = env

    W = L = D = 0

    for i in range(games):
        opponent = opponent_factory(seed + i)

        # Alternate starting side to reduce first-player bias
        agent_is_p1 = (i % 2 == 0)

        result = play_one_game(env, agent, opponent, agent_is_p1)
        if result == "W":
            W += 1
        elif result == "L":
            L += 1
        else:
            D += 1

    total = games
    return EvalResult(
        wins=W,
        losses=L,
        draws=D,
        win_rate=W / total,
        draw_rate=D / total,
        loss_rate=L / total,
    )


def mean_std(values: List[float]):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


# ----------------------------
# Main overfitting test
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Test Connect-4 Q-learning overfitting/generalization")
    parser.add_argument("--qfile", type=str, default="q_table.pkl", help="Path to saved Q-table")
    parser.add_argument("--games", type=int, default=300, help="Games per seed per opponent")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=0.15,
        help="Generalization gap threshold for warning",
    )
    args = parser.parse_args()

    # Load trained agent
    env = EnvConnect4()
    agent = PolicyQLearningV4(env, seed=42)
    agent.load(args.qfile)
    agent.epsilon = 0.0

    # Opponents similar to training
    train_like = {
        "random": lambda s: PolicyRandom(),
        "heuristic": lambda s: PolicyHeuristic(seed=s),
    }

    # Held-out opponents not used in training
    held_out = {
        "center_first": lambda s: PolicyCenterFirst(seed=s),
        "block_then_center": lambda s: PolicyBlockThenCenter(seed=s),
    }

    print("\n=== OVERFITTING / GENERALIZATION TEST ===")
    print(f"Q-table file: {args.qfile}")
    print(f"Games per seed: {args.games}")
    print(f"Number of seeds: {args.seeds}")
    print(f"Gap warning threshold: {args.gap_threshold:.2f}\n")

    summary: Dict[str, List[float]] = {}

    # Training-like opponents
    print("--- TRAIN-LIKE OPPONENTS ---")
    for opp_name, opp_factory in train_like.items():
        seed_win_rates = []
        for seed in range(100, 100 + args.seeds):
            res = evaluate_matchup(agent, opp_factory, games=args.games, seed=seed)
            seed_win_rates.append(res.win_rate)

        m, s = mean_std(seed_win_rates)
        summary[opp_name] = seed_win_rates
        print(f"{opp_name:>18}: mean win rate = {m:.3f} | std = {s:.3f}")
    print()

    # Held-out opponents
    print("--- HELD-OUT OPPONENTS ---")
    for opp_name, opp_factory in held_out.items():
        seed_win_rates = []
        for seed in range(200, 200 + args.seeds):
            res = evaluate_matchup(agent, opp_factory, games=args.games, seed=seed)
            seed_win_rates.append(res.win_rate)

        m, s = mean_std(seed_win_rates)
        summary[opp_name] = seed_win_rates
        print(f"{opp_name:>18}: mean win rate = {m:.3f} | std = {s:.3f}")
    print()

    # Group averages
    train_like_rates = []
    for name in train_like:
        train_like_rates.extend(summary[name])

    held_out_rates = []
    for name in held_out:
        held_out_rates.extend(summary[name])

    train_mean, train_std = mean_std(train_like_rates)
    held_mean, held_std = mean_std(held_out_rates)
    gap = train_mean - held_mean

    print("--- SUMMARY ---")
    print(f"Train-like mean win rate : {train_mean:.3f} ± {train_std:.3f}")
    print(f"Held-out mean win rate   : {held_mean:.3f} ± {held_std:.3f}")
    print(f"Generalization gap       : {gap:.3f}")

    # Also check worst individual opponent
    all_means = {}
    for name, vals in summary.items():
        m, _ = mean_std(vals)
        all_means[name] = m

    worst_opp = min(all_means, key=all_means.get)
    worst_wr = all_means[worst_opp]

    if worst_wr < 0.10:
        print("\nWARNING: Severe weakness detected against at least one opponent.")
        print(f"Worst opponent: {worst_opp} with mean win rate {worst_wr:.3f}")
        print("This suggests poor robustness, even if the average generalization gap looks acceptable.")
    elif gap > args.gap_threshold:
        print("\nWARNING: Large performance gap detected.")
        print("This suggests overfitting to the training opponent distribution.")
    else:
        print("\nGOOD: No major gap or severe opponent-specific weakness detected.")

    print("\nInterpretation:")
    print("- High train-like win rate + much lower held-out win rate => likely overfitting.")
    print("- Similar performance across both groups => better generalization.")
    print("- High std across seeds => unstable policy / sensitive evaluation.")


if __name__ == "__main__":
    main()