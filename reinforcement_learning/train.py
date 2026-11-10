
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reinforcement_learning.environment import (
    ExplorationEnvironment
)

from reinforcement_learning.agent import (
    QLearningAgent
)


def train():

    random.seed(42)
    np.random.seed(42)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    environment = ExplorationEnvironment()

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997
    )

    episodes = 1200

    rewards = []
    coverage = []
    collisions = []
    goals_reached = []

    print("\n======================================")
    print(" FRONTIER-BASED Q-LEARNING TRAINING")
    print("======================================")
    print(f"Episodes: {episodes}")
    print(f"Traversable grid cells: {environment.true_free_cells}")

    for episode in range(episodes):

        state = environment.reset()

        total_reward = 0
        done = False

        while not done:

            action = agent.choose_action(state)

            next_state, reward, done, info = (
                environment.step(action)
            )

            agent.learn(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state
            total_reward += reward

        agent.decay_exploration()

        rewards.append(total_reward)
        coverage.append(info["coverage"])
        collisions.append(info["collisions"])
        goals_reached.append(info["goals_reached"])

        if (episode + 1) % 100 == 0:

            avg_reward = np.mean(rewards[-100:])
            avg_coverage = np.mean(coverage[-100:])
            avg_collisions = np.mean(collisions[-100:])
            avg_goals = np.mean(goals_reached[-100:])

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {avg_reward:.2f} | "
                f"Coverage: {avg_coverage:.2f}% | "
                f"Collisions: {avg_collisions:.2f} | "
                f"Goals: {avg_goals:.2f} | "
                f"States: {len(agent.q_table)} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save(
        "models/frontier_q_table.pkl"
    )

    results = pd.DataFrame({

        "Episode": range(1, episodes + 1),
        "Reward": rewards,
        "Coverage_Percent": coverage,
        "Collisions": collisions,
        "Frontier_Goals_Reached": goals_reached

    })

    results.to_csv(
        "results/frontier_q_learning_training.csv",
        index=False
    )

    # Reward graph
    plt.figure(figsize=(12, 6))

    plt.plot(
        rewards,
        alpha=0.25,
        label="Episode reward"
    )

    plt.plot(
        pd.Series(rewards).rolling(50).mean(),
        linewidth=2,
        label="50-episode average"
    )

    plt.title("Frontier Q-learning Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "results/frontier_q_learning_rewards.png"
    )

    plt.close()

    # Coverage graph
    plt.figure(figsize=(12, 6))

    plt.plot(
        coverage,
        alpha=0.3,
        label="Episode coverage"
    )

    plt.plot(
        pd.Series(coverage).rolling(50).mean(),
        linewidth=2,
        label="50-episode average"
    )

    plt.title("Exploration Coverage During Training")
    plt.xlabel("Episode")
    plt.ylabel("Coverage (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "results/frontier_q_learning_coverage.png"
    )

    plt.close()

    # Frontier goals graph
    plt.figure(figsize=(12, 6))

    plt.plot(
        pd.Series(goals_reached).rolling(50).mean(),
        linewidth=2
    )

    plt.title("Frontier Goals Reached")
    plt.xlabel("Episode")
    plt.ylabel("Average goals reached (50 episodes)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "results/frontier_goals.png"
    )

    plt.close()

    print("\nTraining completed.")
    print("Model: models/frontier_q_table.pkl")
    print("Results: results/frontier_q_learning_training.csv")
    print("Graphs saved in results/")


if __name__ == "__main__":
    train()
