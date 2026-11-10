
import random
import csv
import os
import statistics

from reinforcement_learning.environment import ExplorationEnvironment
from navigation.autonomous_controller import AutonomousController


NUM_SEEDS = 30
ALGORITHMS = ["bfs", "astar"]


def run_episode(algorithm, seed):

    random.seed(seed)

    env = ExplorationEnvironment()
    controller = AutonomousController(algorithm=algorithm)

    done = False

    while not done:

        action = controller.get_action(env)

        if action is None:
            break

        _, reward, done, info = env.step(action)

    status = controller.get_status()

    visited_cells = len(env.visited_cells)
    total_cells = env.grid_rows * env.grid_cols

    coverage = (visited_cells / total_cells) * 100

    return {
        "algorithm": algorithm.upper(),
        "seed": seed,
        "coverage": coverage,
        "visited_cells": visited_cells,
        "collisions": env.collisions,
        "frontiers_reached": status["frontiers_reached"],
        "steps": env.steps,
        "replans": status["replans"],
    }


def calculate_statistics(results, metric):

    values = [result[metric] for result in results]

    average = statistics.mean(values)

    std_dev = (
        statistics.stdev(values)
        if len(values) > 1
        else 0
    )

    return average, std_dev


def main():

    all_results = []

    for algorithm in ALGORITHMS:

        print("\nRunning {} experiments...".format(
            algorithm.upper()
        ))

        algorithm_results = []

        for seed in range(42, 42 + NUM_SEEDS):

            result = run_episode(algorithm, seed)

            algorithm_results.append(result)
            all_results.append(result)

            print(
                "Seed {} | Coverage: {:.2f}% | "
                "Collisions: {} | Frontiers: {}".format(
                    seed,
                    result["coverage"],
                    result["collisions"],
                    result["frontiers_reached"]
                )
            )

        print("{} experiments completed.".format(
            algorithm.upper()
        ))

    print("\n========== FINAL 30-SEED COMPARISON ==========")

    metrics = [
        ("coverage", "Coverage (%)"),
        ("collisions", "Collisions"),
        ("frontiers_reached", "Frontiers reached"),
        ("steps", "Steps"),
        ("replans", "Replans"),
    ]

    for algorithm in ALGORITHMS:

        results = [
            result for result in all_results
            if result["algorithm"] == algorithm.upper()
        ]

        print("\n{}".format(algorithm.upper()))

        for metric, label in metrics:

            average, std_dev = calculate_statistics(
                results,
                metric
            )

            print(
                "{}: {:.2f} +/- {:.2f}".format(
                    label,
                    average,
                    std_dev
                )
            )

    # Save individual experiment results.
    os.makedirs("results", exist_ok=True)

    output_file = "results/planner_comparison.csv"

    with open(output_file, "w", newline="") as file:

        fieldnames = [
            "algorithm",
            "seed",
            "coverage",
            "visited_cells",
            "collisions",
            "frontiers_reached",
            "steps",
            "replans",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nIndividual results saved to:", output_file)


if __name__ == "__main__":
    main()

