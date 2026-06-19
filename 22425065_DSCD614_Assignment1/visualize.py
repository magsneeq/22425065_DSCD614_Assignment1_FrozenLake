"""
visualize.py

Bonus Task - Option B: Visualize training performance using graphs.

Produces:
  results/training_performance.png   - reward / success-rate / epsilon curves
                                        for the best configuration
  results/hyperparameter_comparison.png - success-rate curves for all
                                        experiment configurations, for
                                        comparing learning rate / discount
                                        factor / epsilon decay choices
  results/learned_policy.png         - the learned policy drawn on the grid

Run (after train.py has produced results/stats_*.json and qtable_*.npy):
    python visualize.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from environment import FrozenLakeEnv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COLORS = {
    "reward": "#7b3f8f",
    "success": "#2f7a4f",
    "epsilon": "#b5651d",
    "baseline": "#7b3f8f",
    "high_alpha": "#c0392b",
    "low_gamma": "#2f7a4f",
    "fast_decay": "#2980b9",
}


def moving_average(x, window=200):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_training_performance(window=200):
    with open(os.path.join(RESULTS_DIR, "stats_best.json")) as f:
        stats = json.load(f)

    rewards = stats["episode_rewards"]
    successes = stats["episode_successes"]
    epsilons = stats["epsilon_history"]

    smoothed_rewards = moving_average(rewards, window)
    smoothed_success = moving_average(successes, window) * 100

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))

    axes[0].plot(smoothed_rewards, color=COLORS["reward"])
    axes[0].set_title(f"Episode Reward (moving average, window={window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(smoothed_success, color=COLORS["success"])
    axes[1].set_title(f"Success Rate (moving average, window={window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epsilons, color=COLORS["epsilon"])
    axes[2].set_title("Epsilon Decay Over Training")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "training_performance.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_hyperparameter_comparison(window=200):
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json")) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in summary:
        name = cfg["name"]
        with open(os.path.join(RESULTS_DIR, f"stats_{name}.json")) as f:
            stats = json.load(f)
        smoothed = moving_average(stats["episode_successes"], window) * 100
        label = f"{name} (\u03b1={cfg['alpha']}, \u03b3={cfg['gamma']}, decay={cfg['epsilon_decay']})"
        ax.plot(smoothed, label=label, color=COLORS.get(name))

    ax.set_title(f"Success Rate by Hyperparameter Configuration (moving average, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "hyperparameter_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_learned_policy():
    env = FrozenLakeEnv()
    q_table = np.load(os.path.join(RESULTS_DIR, "qtable_best.npy"))
    policy = np.argmax(q_table, axis=1)

    n_rows, n_cols = env.n_rows, env.n_cols
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for r in range(n_rows):
        for c in range(n_cols):
            s = r * n_cols + c
            if s == env.goal_state:
                color = "#cfe8d5"
            elif s in env.holes:
                color = "#2b2b2b"
            elif s == env.start_state:
                color = "#fde9c8"
            else:
                color = "#eef3f8"
            ax.add_patch(patches.Rectangle((c, n_rows - 1 - r), 1, 1,
                                            facecolor=color, edgecolor="#888888", linewidth=0.8))

            cx, cy = c + 0.5, n_rows - 1 - r + 0.5
            if s == env.goal_state:
                ax.text(cx, cy, "G", ha="center", va="center", fontsize=16, fontweight="bold", color="#2f7a4f")
            elif s in env.holes:
                ax.text(cx, cy, "H", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
            else:
                arrow = env.ACTION_ARROWS[int(policy[s])]
                ax.text(cx, cy, arrow, ha="center", va="center", fontsize=16, color="#5b2c83")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("Learned Greedy Policy")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "learned_policy.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_training_performance()
    plot_hyperparameter_comparison()
    plot_learned_policy()


if __name__ == "__main__":
    main()
