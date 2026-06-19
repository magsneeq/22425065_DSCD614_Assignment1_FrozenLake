"""
agent.py

A from-scratch tabular Q-Learning agent.

Update rule (implemented exactly as specified in the assignment):
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration and decay."""

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=None,
    ):
        """
        Parameters
        ----------
        n_states : int
            Number of discrete states (64 for the 8x8 map).
        n_actions : int
            Number of discrete actions (4: Left, Down, Right, Up).
        alpha : float
            Learning rate (step size for Q-table updates).
        gamma : float
            Discount factor for future rewards.
        epsilon : float
            Initial exploration probability for epsilon-greedy action choice.
        epsilon_min : float
            Floor value epsilon decays towards.
        epsilon_decay : float
            Multiplicative decay applied to epsilon after every episode.
        seed : int or None
            Seed for the agent's RNG, for reproducibility.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-table initialization: all zeros is a common, neutral choice for
        # Frozen Lake since rewards are small/sparse and we don't want to
        # bias the agent towards any action before it has learned anything.
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def choose_action(self, state, greedy=False):
        """
        Epsilon-greedy action selection.

        If greedy=True (used at evaluation time), always pick the action
        with the highest Q-value, ignoring epsilon.
        """
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Apply one Q-Learning update for the observed transition."""
        best_next_value = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self):
        """Geometrically decay epsilon, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """Return the greedy policy (best action per state) derived from the Q-table."""
        return np.argmax(self.q_table, axis=1)

    def get_state_value(self, state):
        """Return V(s) = max_a Q(s, a)."""
        return float(np.max(self.q_table[state]))

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)
        return self.q_table
