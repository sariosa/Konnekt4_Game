import numpy as np
from typing import Dict, Tuple


class PolicyRandom:
    """
    Random policy for Connect 4.

    This policy selects uniformly at random from the currently
    legal actions.
    """

    def __init__(self):
        """
        Initializes the random policy.
        """
        pass

    def _get_action(self, env, observation=None):
        """
        Selects a random legal action.

        Parameters
        ----------
        env : EnvConnect4
            Connect 4 environment.
        observation : dict, optional
            Current observation. Not needed for this policy.

        Returns
        -------
        int
            Selected action as a column index.
        """
        legal = env._get_legal_actions()
        return int(env.np_random.choice(legal))


class PolicyQLearning:
    """
    Q-learning policy for Connect 4.

    The policy stores action values for state-action pairs and
    follows an epsilon-greedy action selection rule.
    """

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, seed: int = 7):
        """
        Initializes the Q-learning policy.

        Parameters
        ----------
        env : EnvConnect4
            Connect 4 environment.
        alpha : float, optional
            Learning rate.
        gamma : float, optional
            Discount factor.
        epsilon : float, optional
            Exploration rate.
        seed : int, optional
            Random seed used for action selection.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Maps a state key to a vector of action values
        self.Q: Dict[Tuple[int, ...], np.ndarray] = {}
        self.n_actions = env.action_space.n

    def _state_key(self, observation):
        """
        Converts an observation into a hashable state key.

        Parameters
        ----------
        observation : dict
            Observation containing the board and the current turn.

        Returns
        -------
        tuple[int, ...]
            Hashable representation of the state.
        """
        return tuple(observation["board"]) + (int(observation["turn"]),)

    def _ensure(self, s: Tuple[int, ...]):
        """
        Ensures that a state exists in the Q-table.

        Parameters
        ----------
        s : tuple[int, ...]
            State key.
        """
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions, dtype=np.float32)

    def _get_action(self, env, observation):
        """
        Selects an action using the epsilon-greedy rule.

        With probability epsilon, the policy explores by choosing
        a random legal action. Otherwise, it exploits by choosing
        the legal action with the highest Q-value.

        Parameters
        ----------
        env : EnvConnect4
            Connect 4 environment.
        observation : dict
            Current observation.

        Returns
        -------
        int
            Selected action as a column index.
        """
        s = self._state_key(observation)
        self._ensure(s)

        legal = env._get_legal_actions()

        # Exploration step
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))

        # Exploitation step with illegal actions masked out
        q = self.Q[s].copy()
        for a in range(self.n_actions):
            if a not in legal:
                q[a] = -np.inf
        return int(np.argmax(q))

    def update(self, s, a, r, s_next, legal_next, done: bool):
        """
        Updates the Q-value for a state-action pair.

        Parameters
        ----------
        s : tuple[int, ...]
            Current state.
        a : int
            Action taken in the current state.
        r : float
            Reward received.
        s_next : tuple[int, ...]
            Next state.
        legal_next : list[int]
            Legal actions in the next state.
        done : bool
            Whether the next state is terminal.
        """
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


class PolicyHeuristic:
    """
    Heuristic policy for Connect 4.

    The decision logic follows four steps:
    1. Play an immediate winning move if available
    2. Block an opponent's immediate winning move if necessary
    3. Prefer columns closer to the center
    4. Fall back to a random legal move
    """

    def __init__(self, seed: int = 42):
        """
        Initializes the heuristic policy.

        Parameters
        ----------
        seed : int, optional
            Random seed used to break ties between equally good moves.
        """
        self.rng = np.random.default_rng(seed)

    def _get_action(self, env, observation):
        """
        Selects an action according to the heuristic rules.

        Parameters
        ----------
        env : EnvConnect4
            Connect 4 environment.
        observation : dict
            Current observation.

        Returns
        -------
        int
            Selected action as a column index.
        """
        legal = env._get_legal_actions()
        player = observation["turn"]
        opponent = 2 if player == 1 else 1

        # Step 1: check for an immediate winning move
        winning_moves = []
        for col in legal:
            row = env._get_drop_row(col)
            idx = env._idx(row, col)

            env.board[idx] = player
            if env.is_winner(player):
                winning_moves.append(col)
            env.board[idx] = 0

        if winning_moves:
            return int(self.rng.choice(winning_moves))

        # Step 2: check whether the opponent has an immediate winning move
        blocking_moves = []
        for col in legal:
            row = env._get_drop_row(col)
            idx = env._idx(row, col)

            env.board[idx] = opponent
            if env.is_winner(opponent):
                blocking_moves.append(col)
            env.board[idx] = 0

        if blocking_moves:
            return int(self.rng.choice(blocking_moves))

        # Step 3: prefer moves closer to the center column
        center = env.num_cols // 2
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

        # Step 4: random fallback
        return int(self.rng.choice(legal))
