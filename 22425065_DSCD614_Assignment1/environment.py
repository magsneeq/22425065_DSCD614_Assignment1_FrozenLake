"""
environment.py

A from-scratch implementation of the Frozen Lake grid-world environment.
No Gymnasium / OpenAI Gym / Stable Baselines / RLlib is used anywhere here -
the environment dynamics, state bookkeeping, and rendering are all
hand-written.

Action encoding (fixed by the assignment spec):
    0 = Left
    1 = Down
    2 = Right
    3 = Up

State representation:
    A single integer index in [0, n_rows * n_cols - 1], computed as
        state = row * n_cols + col
    Coordinates can be recovered with divmod(state, n_cols).
"""

import numpy as np


class FrozenLakeEnv:
    """A custom, dependency-free Frozen Lake environment."""

    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
    ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    ACTION_ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}

    # (delta_row, delta_col) for each action
    ACTION_DELTAS = {
        0: (0, -1),   # Left
        1: (1, 0),    # Down
        2: (0, 1),    # Right
        3: (-1, 0),   # Up
    }

    # The standard 8x8 map given in the assignment brief
    DEFAULT_MAP = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFHFFFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]

    def __init__(
        self,
        grid_map=None,
        step_penalty=-0.01,
        hole_penalty=-1.0,
        goal_reward=1.0,
        slip_prob=0.0,
        seed=None,
    ):
        """
        Parameters
        ----------
        grid_map : list[str] or None
            Rows of the grid map using S/F/H/G characters. Defaults to the
            standard 8x8 map from the assignment.
        step_penalty : float
            Reward given for any non-terminal move (small negative value
            encourages the agent to find short paths rather than wandering
            indefinitely).
        hole_penalty : float
            Reward given for falling into a hole.
        goal_reward : float
            Reward given for reaching the goal.
        slip_prob : float
            Probability that the executed action differs from the intended
            action (stochastic ice). 0.0 reproduces the deterministic
            assignment requirements; set > 0 for the Bonus Option A
            extension.
        seed : int or None
            Seed for the environment's internal RNG (used for slip noise).
        """
        self.grid_map = list(grid_map) if grid_map is not None else list(self.DEFAULT_MAP)
        self.n_rows = len(self.grid_map)
        self.n_cols = len(self.grid_map[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.step_penalty = step_penalty
        self.hole_penalty = hole_penalty
        self.goal_reward = goal_reward
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(seed)

        self.start_state = None
        self.holes = set()
        self.goal_state = None
        self._parse_map()

        self.state = self.start_state
        self.done = False
        self.step_count = 0

    # ------------------------------------------------------------------
    # Map parsing / coordinate helpers
    # ------------------------------------------------------------------
    def _parse_map(self):
        for r, row in enumerate(self.grid_map):
            if len(row) != self.n_cols:
                raise ValueError("All rows in grid_map must have equal length.")
            for c, ch in enumerate(row):
                s = self._to_index(r, c)
                if ch == "S":
                    self.start_state = s
                elif ch == "H":
                    self.holes.add(s)
                elif ch == "G":
                    self.goal_state = s
                elif ch != "F":
                    raise ValueError(f"Unknown tile character '{ch}' in grid_map.")
        if self.start_state is None or self.goal_state is None:
            raise ValueError("grid_map must contain exactly one 'S' and one 'G'.")

    def _to_index(self, row, col):
        return row * self.n_cols + col

    def _to_coords(self, state):
        return divmod(state, self.n_cols)

    # ------------------------------------------------------------------
    # Core API required by the assignment
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the agent to the start state and return it."""
        self.state = self.start_state
        self.done = False
        self.step_count = 0
        return self.state

    def step(self, action):
        """
        Execute one action in the environment.

        Returns
        -------
        next_state : int
        reward : float
        done : bool
        info : dict
        """
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}; must be one of 0,1,2,3.")
        if self.done:
            raise RuntimeError("step() called after episode termination; call reset() first.")

        executed_action = self._apply_slip(action)
        row, col = self._to_coords(self.state)
        d_row, d_col = self.ACTION_DELTAS[executed_action]

        # Enforce movement boundaries: clip to stay on the grid instead of
        # wrapping or moving off the edge.
        new_row = min(max(row + d_row, 0), self.n_rows - 1)
        new_col = min(max(col + d_col, 0), self.n_cols - 1)
        next_state = self._to_index(new_row, new_col)

        self.state = next_state
        self.step_count += 1

        if next_state == self.goal_state:
            reward = self.goal_reward
            self.done = True
        elif next_state in self.holes:
            reward = self.hole_penalty
            self.done = True
        else:
            reward = self.step_penalty
            self.done = False

        info = {
            "intended_action": action,
            "executed_action": executed_action,
            "step_count": self.step_count,
        }
        return next_state, reward, self.done, info

    def render(self, mode="human"):
        """Print (and return) an ASCII picture of the grid with the agent's position."""
        rows = []
        for r in range(self.n_rows):
            chars = []
            for c in range(self.n_cols):
                s = self._to_index(r, c)
                if s == self.state:
                    chars.append("A")  # Agent's current position
                elif s == self.goal_state:
                    chars.append("G")
                elif s in self.holes:
                    chars.append("H")
                elif s == self.start_state:
                    chars.append("S")
                else:
                    chars.append("F")
            rows.append(" ".join(chars))
        grid_str = "\n".join(rows)
        if mode == "human":
            print(grid_str)
        return grid_str

    def get_state(self):
        """Return the agent's current state index."""
        return self.state

    def is_terminal(self, state=None):
        """Return True if the given (or current) state is a hole or the goal."""
        s = self.state if state is None else state
        return s == self.goal_state or s in self.holes

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------
    def get_valid_actions(self, state=None):
        """All four actions are always legal; boundaries simply clip movement."""
        return [0, 1, 2, 3]

    def _apply_slip(self, action):
        """Bonus Option A hook: with probability slip_prob, substitute a random
        other action for the intended one (icy/stochastic transitions)."""
        if self.slip_prob <= 0.0:
            return action
        if self.rng.random() < self.slip_prob:
            other_actions = [a for a in range(self.n_actions) if a != action]
            return int(self.rng.choice(other_actions))
        return action

    def __repr__(self):
        return (
            f"FrozenLakeEnv(n_rows={self.n_rows}, n_cols={self.n_cols}, "
            f"start={self.start_state}, goal={self.goal_state}, "
            f"n_holes={len(self.holes)}, slip_prob={self.slip_prob})"
        )


if __name__ == "__main__":
    # Tiny smoke test / demo
    env = FrozenLakeEnv()
    print(env)
    state = env.reset()
    print("Initial render:")
    env.render()
    print("\nTaking action RIGHT:")
    next_state, reward, done, info = env.step(FrozenLakeEnv.RIGHT)
    env.render()
    print(f"state={next_state}, reward={reward}, done={done}, info={info}")
