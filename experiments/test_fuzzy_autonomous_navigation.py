
import random

import numpy as np

from reinforcement_learning.environment import ExplorationEnvironment
from navigation.autonomous_controller import AutonomousController


def run_episode(algorithm, seed, max_steps=150):

    random.seed(seed)
    np.random.seed(seed)

    env = ExplorationEnvironment()
    controller = AutonomousController(
        algorithm=algorithm
    )

    for step in range(max_steps):

        action = controller.get_action(env)

        if action is None:
            print(
                f"{algorithm.upper()} | Seed {seed}: "
                "No reachable frontier."
            )
            break

        if action not in (0, 1, 2, 3):
            raise AssertionError(
                f"Invalid action returned: {action}"
            )

        _, _, done, _ = env.step(action)

        if done:
            break

    status = controller.get_status()

    return {
        "algorithm": algorithm.upper(),
        "seed": seed,
        "steps": env.steps,
        "visited_cells": len(env.visited_cells),
        "collisions": env.collisions,
        "frontiers_reached": status["frontiers_reached"],
        "safety_checks": status["safety_checks"],
        "safety_overrides": status["safety_overrides"],
        "replans": status["replans"]
    }


def main():

    print("\n========== FUZZY AUTONOMOUS NAVIGATION TEST ==========\n")

    results = []

    for algorithm in ("bfs", "astar"):

        for seed in (42, 43, 44):

            result = run_episode(
                algorithm=algorithm,
                seed=seed
            )

            results.append(result)

            print(
                f"{result['algorithm']} | "
                f"Seed: {result['seed']} | "
                f"Steps: {result['steps']} | "
                f"Visited cells: {result['visited_cells']} | "
                f"Collisions: {result['collisions']} | "
                f"Frontiers: {result['frontiers_reached']} | "
                f"Safety checks: {result['safety_checks']} | "
                f"Overrides: {result['safety_overrides']} | "
                f"Replans: {result['replans']}"
            )

    print("\n========== SUMMARY ==========\n")

    total_steps = sum(
        result["steps"] for result in results
    )

    total_collisions = sum(
        result["collisions"] for result in results
    )

    total_checks = sum(
        result["safety_checks"] for result in results
    )

    total_overrides = sum(
        result["safety_overrides"] for result in results
    )

    print(f"Episodes completed: {len(results)}")
    print(f"Total steps: {total_steps}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total safety checks: {total_checks}")
    print(f"Total safety overrides: {total_overrides}")

    if total_checks > 0:

        override_percentage = (
            total_overrides / total_checks
        ) * 100

        print(
            f"Safety override percentage: "
            f"{override_percentage:.2f}%"
        )

    print("\nIntegration test completed successfully.")


if __name__ == "__main__":
    main()
