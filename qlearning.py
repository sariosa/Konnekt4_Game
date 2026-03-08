# Use with train_qlearning.py for training and evaluation via CLI:
#   python train_qlearning.py --train
#   python train_qlearning.py --train --episodes 200000
#   python train_qlearning.py --eval
#
# The train() function is also importable by play.py as before:
#   from train_qlearning import train
#   agent = train(episodes=50000)

import argparse                      # CLI argument handling
import random                        # Random opponent mixing + seeding
import pickle                        # Saving/loading Q-table
from typing import Optional, List, Tuple  # Type hints for clarity
from dataclasses import dataclass    # Parameter container

import numpy as np                   # Efficient board/feature ops
import matplotlib.pyplot as plt      # Learning curve visualization

from connect4_env import EnvConnect4                     # Connect-4 environment
from policies import PolicyQLearning, PolicyHeuristic, PolicyRandom  # Base policies


# ----------------------------
# Constants
# ----------------------------
ROWS = 6   # Standard Connect-4 board height
COLS = 7   # Standard Connect-4 board width


# ----------------------------
# Feature-based state key (FIX A)
# Replaces raw board tuple — collapses trillions of states into
# ~300k meaningful strategic states the agent can actually learn from.
# ----------------------------
def _canonical_board(board: list) -> tuple:
    """Return left-right canonical form (smaller of board vs mirror)."""
    arr    = np.array(board).reshape(ROWS, COLS)        # Convert to 2D grid
    mirror = np.fliplr(arr).flatten().tolist()          # Mirror across vertical axis
    return tuple(board) if board <= mirror else tuple(mirror)  # Choose canonical symmetric form

def _canonical_action(board: list, action: int) -> int:
    """Mirror action if board was mirrored."""
    arr    = np.array(board).reshape(ROWS, COLS)        # Convert to 2D grid
    mirror = np.fliplr(arr).flatten().tolist()          # Mirror board for comparison
    if board <= mirror:                                 # If original is canonical
        return action                                   # Action unchanged
    return (COLS - 1) - action                           # Mirror the column index

def _col_height(board: list, col: int) -> int:
    """Number of pieces in column (board stored row-major, row 0 = top)."""
    return sum(1 for r in range(ROWS) if board[r * COLS + col] != 0)  # Count non-empty cells in col

def _count_n_in_a_row(board: list, piece: int, n: int) -> int:
    """Count windows of exactly n pieces + (4-n) empties."""
    arr   = np.array(board).reshape(ROWS, COLS)         # 2D board
    empty = 0                                           # Empty cell marker
    count = 0                                           # Total matching windows

    def chk(w):
        return w.count(piece) == n and w.count(empty) == (4 - n)  # Pattern definition (n pieces + rest empty)

    for c in range(COLS - 3):
        for r in range(ROWS):
            if chk([arr[r, c + i] for i in range(4)]): count += 1  # Horizontal windows

    for c in range(COLS):
        for r in range(ROWS - 3):
            if chk([arr[r + i, c] for i in range(4)]): count += 1  # Vertical windows

    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if chk([arr[r + i, c + i] for i in range(4)]): count += 1  # Diagonal ↘ windows

    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if chk([arr[r - i, c + i] for i in range(4)]): count += 1  # Diagonal ↗ windows

    return count                                           # Return number of matching windows

def _immediate_win_cols(env: EnvConnect4, piece: int) -> List[int]:
    """Columns where piece wins immediately."""
    wins = []                                              # Winning columns accumulator
    for col in env._get_legal_actions():                    # Iterate all legal actions
        row = env._get_drop_row(col)                        # Simulated landing row
        if row == -1:
            continue                                        # Skip full columns (safety)
        idx = env._idx(row, col)                            # Convert (row,col) to index
        env.board[idx] = piece                              # Temporarily place piece
        if env.is_winner(piece):
            wins.append(col)                                # Record immediate winning action
        env.board[idx] = 0                                  # Undo simulation
    return wins                                             # Return list of winning columns

def board_to_features(board: list, turn: int, env: EnvConnect4) -> tuple:
    """
    30-feature strategic state key (see feature list in qlearning_v4.py).
    Same board position with irrelevant cell differences → same key.
    """
    cb    = list(_canonical_board(board))                   # Canonicalize board (symmetry)
    piece = turn                                            # Current player piece (1 or 2)
    opp   = 2 if turn == 1 else 1                           # Opponent piece id

    feats = [int(turn)]                                     # Feature 1: whose turn

    # Column heights bucketed (0-3)
    for c in range(COLS):
        feats.append(min(_col_height(cb, c) // 2, 3))       # Coarse height buckets to reduce state variance

    # Agent 2/3-in-a-row (capped)
    feats.append(min(_count_n_in_a_row(cb, piece, 2), 8))   # Count potential 2-in-a-row windows
    feats.append(min(_count_n_in_a_row(cb, piece, 3), 4))   # Count potential 3-in-a-row windows

    # Agent immediate win threats (capped at 3)
    # Temporarily swap env board for counting
    orig = env.board[:]                                     # Backup env board
    env.board = cb[:]                                       # Use canonical board for threat checks
    agent_threats = _immediate_win_cols(env, piece)         # Agent immediate winning moves
    opp_threats   = _immediate_win_cols(env, opp)           # Opponent immediate winning moves
    env.board = orig                                        # Restore env board

    feats.append(min(len(agent_threats), 3))                # Feature: number of agent immediate wins (capped)

    # Opp 2/3-in-a-row (capped)
    feats.append(min(_count_n_in_a_row(cb, opp, 2), 8))      # Opponent 2-in-a-row windows
    feats.append(min(_count_n_in_a_row(cb, opp, 3), 4))      # Opponent 3-in-a-row windows
    feats.append(min(len(opp_threats),   3))                 # Opponent immediate win count (capped)

    # Center column occupancy (col 3)
    center = [cb[r * COLS + 3] for r in range(ROWS) if cb[r * COLS + 3] != 0]  # Extract filled center cells
    if not center:
        feats.append(0)                                     # Center empty
    elif all(p == piece for p in center):
        feats.append(1)                                     # Center controlled by agent
    elif all(p == opp for p in center):
        feats.append(2)                                     # Center controlled by opponent
    else:
        feats.append(3)                                     # Mixed occupancy

    # Per-column win flags
    legal_set = set(env._get_legal_actions() if env.board == board else
                    [c for c in range(COLS) if cb[c] == 0])   # Approx legal cols when using canonical board
    for c in range(COLS):
        feats.append(1 if c in agent_threats else 0)         # Binary flags: agent immediate win in col c
    for c in range(COLS):
        feats.append(1 if c in opp_threats else 0)           # Binary flags: opp immediate win in col c

    # Game phase
    total = sum(1 for x in cb if x != 0)                     # Total stones placed so far
    feats.append(min(total // 4, 10))                        # Coarse phase indicator (early→late)

    return tuple(feats)                                      # Final feature key (hashable)


# ----------------------------
# Improved PolicyQLearning wrapper
# Overrides _state_key to use feature-based representation.
# Everything else (update, _get_action) stays from policies.py.
# ----------------------------
class PolicyQLearningV4(PolicyQLearning):
    """
    Drop-in replacement for PolicyQLearning with feature-based state key.
    Attach env reference so feature extractor can use env helpers.
    """
    def __init__(self, env: EnvConnect4, alpha=0.1, gamma=0.99,
                 epsilon=1.0, seed: int = 42):
        super().__init__(env, alpha=alpha, gamma=gamma,
                         epsilon=epsilon, seed=seed)        # Initialize base Q-learning policy
        self._env_ref = env                                  # Keep env reference for feature extraction

    def _state_key(self, observation):
        board = list(observation["board"])                   # Extract board from observation
        turn  = int(observation["turn"])                     # Extract current turn
        return board_to_features(board, turn, self._env_ref)  # Convert board to compact feature key

    def _get_action(self, env, observation):
        s     = self._state_key(observation)                 # Compute state key (feature-based)
        legal = env._get_legal_actions()                      # Current legal moves
        self._ensure(s)                                       # Ensure Q-table has an entry for state

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))               # Exploration: random legal action

        # Canonical action mapping for symmetry (FIX C)
        board = list(observation["board"])                   # Raw board (for legality masking)
        q     = self.Q[s].copy()                              # Copy Q-values for masking
        for a in range(self.n_actions):
            if a not in legal:
                q[a] = -np.inf                               # Mask illegal actions
        return int(np.argmax(q))                              # Exploitation: best legal action

    def save(self, filename: str = "q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)                     # Persist Q-table to disk
        print(f"Saved Q-table → {filename}  (states={len(self.Q)})")  # Log save summary

    def load(self, filename: str = "q_table.pkl"):
        from collections import defaultdict
        with open(filename, "rb") as f:
            data = pickle.load(f)                            # Load stored Q-table
        self.Q = data                                        # Restore Q-table
        print(f"Loaded Q-table ← {filename}  (states={len(self.Q)})")  # Log load summary


# ----------------------------
# Reward shaping helpers
# ----------------------------
def _has_safe_move(env: EnvConnect4, acting_piece: int, opp_piece: int) -> bool:
    """True if acting_piece has at least one move that doesn't give opp immediate win."""
    for col in env._get_legal_actions():                     # Check all legal moves
        row = env._get_drop_row(col)                         # Landing row for this move
        if row == -1:
            continue                                         # Skip full columns
        idx = env._idx(row, col)                             # Convert to board index
        env.board[idx] = acting_piece                        # Simulate acting player's move
        opp_wins = _immediate_win_cols(env, opp_piece)       # Check if opponent then has immediate win
        env.board[idx] = 0                                   # Undo simulation
        if not opp_wins:
            return True                                      # Found at least one safe action
    return False                                             # No safe action exists


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainParams:
    episodes:      int   = 200_000
    alpha_start:   float = 0.15
    alpha_end:     float = 0.01
    gamma:         float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.999996   
    epsilon_min:   float = 0.05
    heuristic_prob: float = 0.80      
    seed:          int   = 42
    print_every:   int   = 5_000
    q_file:        str   = "q_table.pkl"
    curve_file:    str   = "learning_curve.png"


def train(
    episodes:   int = 200_000,
    seed:       int = 42,
    print_every: int = 5_000,
    q_file:     str = "q_table.pkl",
    curve_file: str = "learning_curve.png",
) -> PolicyQLearningV4:
    """
    Train and return a PolicyQLearningV4 agent.
    Importable by play.py:  agent = train(episodes=200000)
    """
    params = TrainParams(
        episodes=episodes, seed=seed,
        print_every=print_every, q_file=q_file, curve_file=curve_file
    )                                                        # Bundle training configuration

    random.seed(seed)                                        # Seed Python RNG
    np.random.seed(seed)                                     # Seed NumPy RNG

    env        = EnvConnect4()                                # Create environment
    heuristic  = PolicyHeuristic(seed=seed)                   # Heuristic opponent policy
    rand_pol   = PolicyRandom()                               # Random opponent policy

    agent = PolicyQLearningV4(
        env,
        alpha   = params.alpha_start,
        gamma   = params.gamma,
        epsilon = params.epsilon_start,
        seed    = seed,
    )                                                        # Initialize learning agent

    wins = losses = draws = 0                                 # Global outcome counters
    window = 1000                                             # Moving average window size
    recent: List[int]   = []                                  # Store recent win indicators
    curve:  List[float] = []                                  # Store moving win-rate curve

    print(f"Training v4 | episodes={episodes} | ε_decay={params.epsilon_decay} "
          f"| α {params.alpha_start}→{params.alpha_end} "
          f"| heuristic_prob={params.heuristic_prob}")         # Log configuration header

    for ep in range(1, episodes + 1):                         # Loop over episodes

        # FIX F: linear alpha decay
        frac        = ep / episodes                           # Progress fraction (0→1)
        agent.alpha = params.alpha_start + frac * (params.alpha_end - params.alpha_start)  # Decay α

        # FIX B: alternate who goes first
        obs, info = env.reset()                               # Reset environment for new game
        if ep % 2 == 0:
            # Agent goes first — make one agent move before the loop
            pass   # turn starts at 1 already in env          # No-op since env already starts at player 1

        episode_over = False                                   # Episode termination flag
        last_obs     = None                                    # Track last agent state (for opponent win credit)
        last_action  = None                                    # Track last agent action

        while not episode_over:                                # Step through the game
            current_turn = env.turn                            # Current player turn (1 or 2)
            legal        = env._get_legal_actions()             # Legal columns available

            if not legal:
                draws += 1                                     # No legal moves means draw
                break

            # ---- Agent's turn (player 1 always, or player 2 on alternating eps) ----
            agent_turn = (current_turn == 1 and ep % 2 != 0) or \
                         (current_turn == 2 and ep % 2 == 0)    # Decide whether agent controls this turn

            if agent_turn:
                s      = agent._state_key(obs)                 # Current feature-based state key
                action = agent._get_action(env, obs)            # Choose action via ε-greedy policy

                last_obs    = obs                               # Save last agent observation (credit assignment)
                last_action = action                            # Save last agent action

                # --- Reward shaping BEFORE step ---
                piece      = current_turn                       # Acting player id (agent side)
                opp_piece  = 2 if piece == 1 else 1              # Opponent id
                must_block = _immediate_win_cols(env, opp_piece) # Opponent immediate wins to potentially block

                obs_next, reward, terminated, truncated, info_next = env.step(action)  # Apply action in env
                episode_over = terminated or truncated           # Update termination flag

                s_next     = agent._state_key(obs_next)          # Next state key for Q-learning update
                legal_next = info_next["legal_columns"]          # Legal actions from next state

                if terminated:
                    winner = info_next["winner"]                 # Winner id from env
                    if winner == piece:
                        shaped_reward = +5.0                     # FIX H: strong win reward
                        wins += 1
                    elif winner != 0:
                        shaped_reward = -5.0                     # FIX H: strong loss reward (e.g., illegal)
                        losses += 1
                    else:
                        shaped_reward = 0.0                      # Draw reward
                        draws += 1
                else:
                    # Shaping for non-terminal step
                    block_bonus = +2.0 if (must_block and action in must_block) else 0.0  # Reward correct block

                    # Threat penalty only if safe move existed (FIX G)
                    opp_threats_after = _immediate_win_cols(env, opp_piece)               # Opp threats after move
                    if opp_threats_after:
                        threat_penalty = -2.0 if _has_safe_move(env, piece, opp_piece) else 0.0  # Penalize only if avoidable
                    else:
                        threat_penalty = 0.0                                              # No immediate threat → no penalty

                    # Offensive nudge: own threats created
                    own_threats  = _immediate_win_cols(env, piece)                         # Agent threats after move
                    threat_bonus = 0.1 * min(len(own_threats), 3)                          # Small incentive for pressure

                    shaped_reward = -0.01 + block_bonus + threat_penalty + threat_bonus    # Step cost + shaping

                agent.update(s, action, shaped_reward, s_next, legal_next, episode_over)   # Q-learning update

                # FIX E: epsilon decay per step
                if agent.epsilon > params.epsilon_min:
                    agent.epsilon = max(params.epsilon_min,
                                        agent.epsilon * params.epsilon_decay)             # Decay exploration safely

                obs = obs_next                                                             # Advance observation

            else:
                # ---- Opponent's turn ----
                if random.random() < params.heuristic_prob:
                    action = heuristic._get_action(env, obs)                               # Use heuristic opponent
                else:
                    action = rand_pol._get_action(env, obs)                                # Use random opponent

                obs_next, reward, terminated, truncated, info_next = env.step(action)      # Apply opponent action
                episode_over = terminated or truncated                                     # Update termination

                if terminated and info_next["winner"] == current_turn:
                    # Opponent won — punish agent's last action (FIX H)
                    losses += 1
                    if last_obs is not None and last_action is not None:
                        s_last      = agent._state_key(last_obs)                           # State before last agent move
                        s_next_last = agent._state_key(obs_next)                           # Terminal next state
                        agent.update(s_last, last_action, -5.0,
                                     s_next_last, [], True)                                # Terminal loss update
                elif terminated:
                    if info_next["is_draw"]:
                        draws += 1                                                         # Count draws explicitly

                obs = obs_next                                                             # Advance observation

        # Track moving win rate
        recent.append(1 if info_next.get("winner", 0) in
                      ([1] if ep % 2 != 0 else [2]) else 0)                                # Win indicator (agent side)
        if len(recent) > window:
            recent.pop(0)                                                                  # Maintain fixed window
        curve.append(sum(recent) / len(recent))                                            # Append moving win-rate

        if ep % print_every == 0:
            cum_win = 100.0 * wins / ep                                                    # Cumulative win %
            mov_win = 100.0 * curve[-1]                                                    # Moving win % (windowed)
            print(
                f"Ep {ep:>7}/{episodes} | "
                f"CumWin%={cum_win:5.2f} | "
                f"Win(last{window})={mov_win:5.2f}% | "
                f"W/L/D {wins}/{losses}/{draws} | "
                f"ε={agent.epsilon:.4f}  α={agent.alpha:.4f} | "
                f"states={len(agent.Q)}"
            )                                                                              # Log progress snapshot

    print("\nTraining complete.")                                                          # Training end marker
    agent.save(q_file)                                                                     # Save final Q-table

    # Learning curve plot
    fig, ax = plt.subplots(figsize=(12, 5))                                                # Create plot figure/axes
    ax.plot(curve, linewidth=0.6, alpha=0.4, color="steelblue", label="Win rate (raw)")    # Plot raw win-rate curve
    if len(curve) > 5000:
        n        = 5000
        smoothed = np.convolve(curve, np.ones(n) / n, mode="valid")                        # Compute moving smooth curve
        ax.plot(range(n // 2, n // 2 + len(smoothed)), smoothed,
                linewidth=2.5, color="coral", label=f"Smoothed (n={n})")                   # Plot smoothed curve
    ax.set_xlabel("Episode")                                                               # X label
    ax.set_ylabel(f"Win Rate (moving avg, window={window})")                               # Y label
    ax.set_title("Training Learning Curve — Q-learning v4 (Feature-Based States)")         # Title
    ax.legend()                                                                            # Legend
    ax.set_ylim(0, 1)                                                                      # Clamp to [0,1]
    plt.tight_layout()                                                                     # Improve spacing
    plt.savefig(curve_file, dpi=150)                                                       # Save plot image
    print(f"Saved learning curve → {curve_file}")  
    
    evaluate(q_file=q_file, games=1000, seed=seed, out_plot="evaluation.png")                                         # Log saved file path

    return agent                                                                           # Return trained agent


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(
    q_file: str = "q_table.pkl",
    games: int = 1000,
    seed: int = 42,
    out_plot: str = "evaluation.png",
):
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
        print(f"  vs {opp:>10}: W/L/D = {v['W']:>4}/{v['L']:>4}/{v['D']:>4} | win_rate = {v['win_rate']:.3f}")

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
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Saved evaluation plot → {out_plot}")                                         # Log saved file name


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Q-learning v4 for Connect-4")             # CLI parser
    parser.add_argument("--train",       action="store_true")                               # Enable training mode
    parser.add_argument("--eval",        action="store_true")                               # Enable evaluation mode
    parser.add_argument("--episodes",    type=int, default=200_000)                         # Training episodes
    parser.add_argument("--games",       type=int, default=1000)                            # Evaluation games
    parser.add_argument("--seed",        type=int, default=42)                              # Random seed
    parser.add_argument("--print_every", type=int, default=5_000)                           # Logging frequency
    parser.add_argument("--qfile",       type=str, default="q_table.pkl")                   # Q-table filename
    parser.add_argument("--curve",       type=str, default="learning_curve.png")           # Learning curve filename
    args = parser.parse_args()                                                              # Parse CLI args

    if not (args.train or args.eval):
        print("Specify --train and/or --eval")                                              # Require at least one action
        return

    if args.train:
        train(
            episodes=args.episodes, seed=args.seed,
            print_every=args.print_every,
            q_file=args.qfile, curve_file=args.curve,
        )                                                                                   # Run training pipeline
    if args.eval:
        evaluate(q_file=args.qfile, games=args.games, seed=args.seed)                       # Run evaluation pipeline


if __name__ == "__main__":
    main()                                                                                  # Entry point