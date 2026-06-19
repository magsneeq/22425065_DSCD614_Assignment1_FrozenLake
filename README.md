# Frozen Lake from First Principles Using Q-Learning

DSCD 614 - Reinforcement Learning
Name: Margaret Naa Dei Neequaye
ID: 22425065

A complete, dependency-free implementation of the Frozen Lake environment and
a tabular Q-Learning agent. No Gymnasium, OpenAI Gym, Stable Baselines, or
RLlib is used anywhere in this repository - the environment, agent, training
loop, and evaluation pipeline are all implemented from scratch in Python with
only `numpy` and `matplotlib` as dependencies.

## Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a framework in which an agent learns to make
decisions by interacting with an environment. At each time step, the agent
observes a state, takes an action, and receives a reward along with the next
state. Rather than being told the correct action directly (as in supervised
learning), the agent must discover good behaviour through trial and error,
gradually learning a policy - a mapping from states to actions - that
maximizes the cumulative reward it can expect to receive over time.

### What is Frozen Lake?

Frozen Lake is a classic grid-world benchmark for RL. An agent starts at a
fixed cell on a frozen lake and must reach a goal cell while avoiding holes
in the ice. Each non-terminal cell is "frozen" (safe to stand on); stepping
into a hole ends the episode in failure, and reaching the goal ends the
episode in success. The challenge for the agent is to learn, purely from
experience, which sequence of moves reliably leads to the goal while
threading between the holes.

## Environment Design

The environment is implemented in `environment.py` as the `FrozenLakeEnv`
class, using the standard 8x8 map given in the assignment brief:

```
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFHFFFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

### State representation

States are represented as a single integer index in `[0, 63]`, computed as
`state = row * 8 + col`. This is a compact representation that maps directly
onto the rows of the Q-table (one row of the table per state). Coordinates
can always be recovered with `divmod(state, 8)` when needed for rendering.

### Action representation

Actions are encoded as integers, exactly as specified in the assignment:

| Action | Meaning |
|--------|---------|
| 0      | Left    |
| 1      | Down    |
| 2      | Right   |
| 3      | Up      |

Movement is clipped at the grid boundary - if an action would move the agent
off the edge of the grid, the agent simply stays in place along that axis
rather than wrapping around or raising an error.

### Reward structure

The assignment leaves reward design open, so the following structure was
chosen deliberately to make learning both correct and reasonably fast:

| Outcome                  | Reward |
|---------------------------|--------|
| Reaching the goal (`G`)   | `+1.0` |
| Falling into a hole (`H`) | `-1.0` |
| Any other move            | `-0.01`|

The small per-step penalty discourages the agent from wandering aimlessly
once it has learned to avoid holes, while the asymmetric +1 / -1 terminal
rewards give a strong, unambiguous learning signal at episode boundaries.
This is a richer signal than the "0 everywhere except +1 at the goal"
reward scheme used in some Frozen Lake implementations, and it noticeably
speeds up convergence in practice (see Results below).

## Q-Learning Algorithm

### Description of Q-Learning

Q-Learning is a model-free, off-policy temporal-difference algorithm. It
learns a table of action-values, `Q(s, a)`, that estimates the expected
discounted return of taking action `a` in state `s` and then acting
optimally thereafter. Because the algorithm bootstraps from its own current
estimate of the best future action (rather than the action actually taken
next), it can learn a near-optimal policy even while behaving partly
randomly during exploration.

### Explanation of the update equation

After every transition `(s, a, r, s')`, the agent applies:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

- `alpha` (learning rate) controls how much each new transition shifts the
  estimate. The Q-table is updated in `agent.py`'s `update()` method exactly
  in this form.
- `gamma` (discount factor) controls how strongly future rewards are valued
  relative to immediate ones.
- `max_a' Q(s', a')` is the agent's current best estimate of the value of
  the next state, which is what makes the algorithm bootstrap and converge
  toward the true optimal Q-values over repeated visits.
- The bracketed term `[r + gamma * max_a' Q(s', a') - Q(s, a)]` is the
  temporal-difference (TD) error: the gap between what was predicted and
  what was observed.

### Exploration strategy

The agent uses epsilon-greedy exploration: with probability `epsilon` it
picks a uniformly random action, and otherwise picks the action with the
highest current Q-value. `epsilon` starts at `1.0` (fully random, to explore
broadly before the Q-table contains any useful information) and decays
geometrically after every episode (`epsilon <- max(epsilon_min, epsilon *
epsilon_decay)`) down to a floor of `0.01`, so that the agent settles into
exploiting its learned policy as training progresses.

## Training Procedure

Training is driven by `train.py`, which trains the agent for `10,000`
episodes (capped at `200` steps per episode) under four different
hyperparameter configurations, as required by Part C:

| Configuration | alpha | gamma | epsilon decay |
|---------------|-------|-------|----------------|
| baseline      | 0.1   | 0.99  | 0.9995         |
| high_alpha    | 0.5   | 0.99  | 0.9995         |
| low_gamma     | 0.1   | 0.90  | 0.9995         |
| fast_decay    | 0.1   | 0.99  | 0.999          |

For every configuration, per-episode reward, success flag, and the epsilon
value are logged and saved to `results/stats_<name>.json`, and the resulting
Q-table is saved to `results/qtable_<name>.npy`. The best-performing
configuration (by success rate over the final 500 training episodes) is
additionally saved as `results/qtable_best.npy` / `results/stats_best.json`,
and is the one used for evaluation and policy extraction.

## Results

### Final success rate

| Configuration | Final success rate (last 500 episodes) | Final avg. reward |
|---------------|------------------------------------------|--------------------|
| baseline      | 99.60%                                    | 0.8607             |
| high_alpha    | 98.80%                                    | 0.8452             |
| low_gamma     | 99.60%                                    | 0.8609             |
| fast_decay    | 99.00%                                    | 0.8486             |

`baseline` (alpha=0.1, gamma=0.99, epsilon_decay=0.9995) was selected as the
best configuration and is the one reported below; `low_gamma` reached an
essentially identical final success rate.

Evaluating the `baseline` agent's greedy (exploitation-only) policy over
**200** held-out episodes:

| Metric                  | Value   |
|---------------------------|---------|
| Success Rate               | 100.00% |
| Average Reward              | 0.8700  |
| Average Episode Length      | 14.0 steps |
| Successful Runs             | 200     |
| Number of Failures          | 0       |

Because the environment is deterministic (no slip noise), the greedy policy
always takes the identical 14-step path, so every evaluation episode
succeeds.

### Learned policy

```
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
→ → → → ↓ ↓ ↓ ↓
↑ ↑ ↑ H → → ↓ ↓
→ ↑ ↑ H → → → ↓
↑ ↑ ↑ H → → → ↓
↑ H H → → ↑ H ↓
↓ H → ↑ H ↑ H ↓
→ → ↑ H → ↑ → G
```

(`H` = hole, `G` = goal; arrows show the action the policy recommends for
that cell.) See `results/learned_policy.png` for a colour-coded version of
this grid.

### Discussion of performance

All four configurations converge to a high (>98%) success rate, confirming
that the from-scratch Q-Learning implementation is correctly learning the
task. The hyperparameter sweep (visualized in
`results/hyperparameter_comparison.png`) shows clear differences in
*convergence speed* rather than final quality:

- `fast_decay` (faster epsilon decay) converges substantially faster than
  the others, since it spends less time exploring randomly and starts
  exploiting its (already-decent) policy sooner. On a small, easy
  8x8 deterministic grid this pays off; on a harder or stochastic
  environment, decaying too fast risks settling into a suboptimal policy
  before enough exploration has occurred.
- `high_alpha` (alpha=0.5) is noticeably *slower* and noisier to converge
  than the baseline (alpha=0.1) despite having a larger step size. Large
  updates over-write previous estimates aggressively, which increases
  variance in the early/middle training phase.
- `low_gamma` (gamma=0.9) performs almost identically to the baseline
  (gamma=0.99) on this map, since the optimal path is short (14 steps) and
  a discount factor of 0.9 still weighs a reward 14 steps away heavily
  enough to drive the same behaviour.

## Environment Design - Execution Instructions

### Setup

```bash
pip install -r requirements.txt
```

### Run training (Parts B, C, D)

```bash
python train.py
```

This trains all four hyperparameter configurations, prints progress every
2,000 episodes, and writes Q-tables, training statistics, the learned
policy, and an experiment summary to `results/`.

### Run evaluation (Part E)

```bash
python evaluate.py
```

Loads `results/qtable_best.npy` and evaluates the greedy policy over 200
episodes, printing the success rate, average reward, and failure count, and
saving the full results to `results/evaluation_results.json`.

### Generate bonus visualizations (Bonus Option B)

```bash
python visualize.py
```

Produces `results/training_performance.png` (reward / success-rate /
epsilon curves), `results/hyperparameter_comparison.png` (success-rate
curves across all four configurations), and `results/learned_policy.png`
(the learned policy drawn on the grid).

## Repository Structure

```
frozen-lake-qlearning/
├── environment.py        # FrozenLakeEnv (Part A)
├── agent.py               # QLearningAgent (Part B)
├── train.py                # training loop + hyperparameter sweep (Parts C, D)
├── evaluate.py              # policy evaluation (Part E)
├── visualize.py              # Bonus Option B: training graphs
├── requirements.txt
├── README.md
├── report.pdf
└── results/
    ├── qtable_<name>.npy            # Q-table per configuration
    ├── qtable_best.npy               # Q-table for the best configuration
    ├── stats_<name>.json              # per-episode training stats
    ├── stats_best.json
    ├── experiment_summary.json         # final metrics per configuration
    ├── learned_policy.txt               # text rendering of the policy
    ├── evaluation_results.json           # Part E evaluation output
    ├── training_performance.png           # bonus graphs
    ├── hyperparameter_comparison.png
    └── learned_policy.png
```

## Bonus Task Implemented

**Option B - Visualize training performance using graphs.** See
`visualize.py` and the `results/*.png` files described above.
