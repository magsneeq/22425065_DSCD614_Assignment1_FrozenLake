"""
train.py

Trains a Q-Learning agent on the custom FrozenLakeEnv, experimenting with
different learning rates, discount factors, and exploration decay rates,
as required by Part C of the assignment.

Run:
    python train.py
"""

import os
import json
import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def train_agent(env, agent, n_episodes=10000, max_steps=200, verbose_every=2000):
    """
    Run the Q-Learning training loop and collect statistics.

    Returns a dict with per-episode reward, success flag, and the epsilon
    value at the end of each episode.
    """
    episode_rewards = []
    episode_successes = []
    epsilon_history = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        success = False

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                success = (state == env.goal_state)
                break

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_successes.append(1 if success else 0)
        epsilon_history.append(agent.epsilon)

        if verbose_every and (ep + 1) % verbose_every == 0:
            window = min(verbose_every, len(episode_successes))
            recent_success_rate = np.mean(episode_successes[-window:]) * 100
            recent_avg_reward = np.mean(episode_rewards[-window:])
            print(
                f"  Episode {ep + 1:>6}/{n_episodes} | "
                f"success rate (last {window}): {recent_success_rate:6.2f}% | "
                f"avg reward: {recent_avg_reward:7.4f} | epsilon: {agent.epsilon:.4f}"
            )

    return {
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "epsilon_history": epsilon_history,
    }


def grid_to_policy_string(env, policy):
    """Render the greedy policy as an 8x8 grid of arrows / H / G symbols."""
    lines = []
    for r in range(env.n_rows):
        row_symbols = []
        for c in range(env.n_cols):
            s = r * env.n_cols + c
            if s == env.goal_state:
                row_symbols.append("G")
            elif s in env.holes:
                row_symbols.append("H")
            else:
                row_symbols.append(FrozenLakeEnv.ACTION_ARROWS[int(policy[s])])
        lines.append(" ".join(row_symbols))
    return "\n".join(lines)


def run_experiment(name, alpha, gamma, epsilon_decay, n_episodes, seed):
    env = FrozenLakeEnv(seed=seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    print(f"\n=== Experiment '{name}': alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay} ===")
    stats = train_agent(env, agent, n_episodes=n_episodes)

    tail = min(500, n_episodes)
    final_success_rate = float(np.mean(stats["episode_successes"][-tail:]) * 100)
    final_avg_reward = float(np.mean(stats["episode_rewards"][-tail:]))
    print(f"  -> Final success rate (last {tail} episodes): {final_success_rate:.2f}%")

    return env, agent, stats, final_success_rate, final_avg_reward


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_episodes = 10000
    base_seed = 42

    # Hyperparameter configurations explored for Part C.
    experiments = [
        {"name": "baseline",   "alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.9995},
        {"name": "high_alpha", "alpha": 0.5, "gamma": 0.99, "epsilon_decay": 0.9995},
        {"name": "low_gamma",  "alpha": 0.1, "gamma": 0.90, "epsilon_decay": 0.9995},
        {"name": "fast_decay", "alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.999},
    ]

    summary = []
    best = None

    for cfg in experiments:
        env, agent, stats, final_rate, final_reward = run_experiment(
            cfg["name"], cfg["alpha"], cfg["gamma"], cfg["epsilon_decay"],
            n_episodes=n_episodes, seed=base_seed,
        )

        np.save(os.path.join(RESULTS_DIR, f"qtable_{cfg['name']}.npy"), agent.q_table)
        with open(os.path.join(RESULTS_DIR, f"stats_{cfg['name']}.json"), "w") as f:
            json.dump(stats, f)

        record = {**cfg, "final_success_rate": final_rate, "final_avg_reward": final_reward}
        summary.append(record)

        if best is None or final_rate > best["final_success_rate"]:
            best = {**record, "env": env, "agent": agent, "stats": stats}

    # Persist the best-performing configuration as the "main" trained agent
    np.save(os.path.join(RESULTS_DIR, "qtable_best.npy"), best["agent"].q_table)
    with open(os.path.join(RESULTS_DIR, "stats_best.json"), "w") as f:
        json.dump(best["stats"], f)

    policy = best["agent"].get_policy()
    policy_str = grid_to_policy_string(best["env"], policy)

    with open(os.path.join(RESULTS_DIR, "learned_policy.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"Best configuration: {best['name']} "
            f"(alpha={best['alpha']}, gamma={best['gamma']}, "
            f"epsilon_decay={best['epsilon_decay']})\n"
        )
        f.write(f"Final training success rate (last 500 episodes): {best['final_success_rate']:.2f}%\n\n")
        f.write(policy_str + "\n")

    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Hyperparameter experiment summary ===")
    for r in summary:
        print(
            f"  {r['name']:<12} alpha={r['alpha']:<5} gamma={r['gamma']:<5} "
            f"eps_decay={r['epsilon_decay']:<7} -> "
            f"success={r['final_success_rate']:6.2f}%  avg_reward={r['final_avg_reward']:.4f}"
        )

    print(f"\nBest configuration: '{best['name']}' "
          f"with {best['final_success_rate']:.2f}% success rate over the final 500 episodes.")
    print("\nLearned policy (best configuration):")
    print(policy_str)


if __name__ == "__main__":
    main()
