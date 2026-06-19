"""
evaluate.py

Evaluates a trained Q-Learning policy on the FrozenLakeEnv over a fixed
number of episodes using purely greedy (exploitation-only) action choice,
as required by Part E of the assignment.

Run:
    python evaluate.py
"""

import os
import json
import numpy as np

from environment import FrozenLakeEnv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def evaluate_policy(env, q_table, n_episodes=100, max_steps=200):
    """Run n_episodes greedy rollouts and collect summary statistics."""
    successes = 0
    failures = 0
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        for _ in range(max_steps):
            action = int(np.argmax(q_table[state]))
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            state = next_state
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        if state == env.goal_state:
            successes += 1
        else:
            failures += 1

    n = n_episodes
    return {
        "n_episodes": n,
        "success_rate_pct": successes / n * 100,
        "average_reward": float(np.mean(episode_rewards)),
        "average_episode_length": float(np.mean(episode_lengths)),
        "successes": successes,
        "failures": failures,
        "episode_rewards": episode_rewards,
    }


def main():
    qtable_path = os.path.join(RESULTS_DIR, "qtable_best.npy")
    if not os.path.exists(qtable_path):
        raise FileNotFoundError(
            f"Could not find {qtable_path}. Run train.py first to produce a trained Q-table."
        )

    q_table = np.load(qtable_path)
    env = FrozenLakeEnv(seed=123)  # fresh, deterministic env for evaluation

    n_episodes = 200  # comfortably above the assignment's minimum of 100
    results = evaluate_policy(env, q_table, n_episodes=n_episodes)

    print(f"=== Evaluation Results ({n_episodes} episodes, greedy policy) ===")
    print(f"Success Rate:            {results['success_rate_pct']:.2f}%")
    print(f"Average Reward:          {results['average_reward']:.4f}")
    print(f"Average Episode Length:  {results['average_episode_length']:.2f} steps")
    print(f"Successful Runs:         {results['successes']}")
    print(f"Number of Failures:      {results['failures']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
