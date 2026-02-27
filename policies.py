import numpy as np
from typing import Dict, Tuple

# RANDOM POLICY
class PolicyRandom:
    def __init__(self):
        pass

    def _get_action(self, env, observation=None):
        legal = env._get_legal_actions()
        return int(env.np_random.choice(legal))

# Q-LEARNING POLICY
class PolicyQLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, seed: int = 7):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Q[(board..., turn)] -> np.array(num_actions)
        self.Q: Dict[Tuple[int, ...], np.ndarray] = {}
        self.n_actions = env.action_space.n

    def _state_key(self, observation):
        return tuple(observation["board"]) + (int(observation["turn"]),)

    def _ensure(self, s: Tuple[int, ...]):
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions, dtype=np.float32)

    def _get_action(self, env, observation):
        s = self._state_key(observation)
        self._ensure(s)

        legal = env._get_legal_actions()

        # Explore
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))

        # Exploit (mask illegal actions)
        q = self.Q[s].copy()
        for a in range(self.n_actions):
            if a not in legal:
                q[a] = -np.inf
        return int(np.argmax(q))

    def update(self, s, a, r, s_next, legal_next, done: bool):
        self._ensure(s)
        self._ensure(s_next)

        if done:
            target = float(r)
        else:
            q_next = self.Q[s_next].copy()
            for aa in range(self.n_actions):
                if aa not in legal_next:
                    q_next[aa] = -np.inf
            target = float(r) + self.gamma * float(np.max(q_next))

        self.Q[s][a] = self.Q[s][a] + self.alpha * (target - self.Q[s][a])

# HEURISTIC POLICY
class PolicyHeuristic:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _get_action(self, env, observation):
        legal = env._get_legal_actions()
        player = observation["turn"]
        opponent = 2 if player == 1 else 1
        
        # 1) Immediate Win
        winning_moves = []
        for col in legal:
            row = env._get_drop_row(col)
            idx = env._idx(row, col)

            env.board[idx] = player
            if env.is_winner(player):
                winning_moves.append(col)
            env.board[idx] = 0  # Undo move

        if winning_moves:
            return int(self.rng.choice(winning_moves))

      
        # 2) Immediate Block
        blocking_moves = []
        for col in legal:
            row = env._get_drop_row(col)
            idx = env._idx(row, col)

            env.board[idx] = opponent
            if env.is_winner(opponent):
                blocking_moves.append(col)
            env.board[idx] = 0  # Undo move

        if blocking_moves:
            return int(self.rng.choice(blocking_moves))

        # 3) Center Preference
        center = env.num_cols // 2  # column 3
        best_score = -float("inf")
        best_moves = []

        for col in legal:
            score = -abs(col - center)
            if score > best_score:
                best_score = score
                best_moves = [col]
            elif score == best_score:
                best_moves.append(col)

        if best_moves:
            return int(self.rng.choice(best_moves))

        # 4) Fallback (should rarely trigger)
        return int(self.rng.choice(legal))
