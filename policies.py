import numpy as np
from typing import Dict, Tuple


class PolicyRandom:
    def __init__(self):
        pass

    def _get_action(self, env, observation=None):
        legal = env._get_legal_actions()
        return int(env.np_random.choice(legal))


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

        # explore
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))

        # exploit (mask illegal)
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