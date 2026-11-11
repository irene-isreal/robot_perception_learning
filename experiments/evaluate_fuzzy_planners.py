
import random
import csv
import os
import statistics

from reinforcement_learning.environment import ExplorationEnvironment
from navigation.autonomous_controller import AutonomousController


NUM_SEEDS = 30
START_SEED = 42
ALGORITHMS = ["bfs", "astar"]


def get_true_free_cells(env):
    """
    Return all grid cells whose centres are valid robot positions.
    Cell coordinates use (gx, gy), matching env.visited_cells.
    """

    true_free_cells = set()

    for gy in range(env.grid_rows):
        for gx in range(env.grid_cols):

            x, y = env.grid_to_world(gx, gy)

            if env.is_valid_position(x, y):
                true_free_cells.add((gx, gy))

    return true_free_cells


def run_episode(algorithm, seed):

    random.seed(seed)

    env = ExplorationEnvironment()
    controller = AutonomousController(algorithm=algorithm)

    true_free_cells = get_true_free_cells(env)

    done = False

    while not done:

        action = controller.get_action(env)

        if action is None:
            break

        _, reward, done, info = env.step(action)

    status = controller.get_status()

    visited_cells = len(env.visited_cells)

    visited_free_cells = len(
        env.visited_cells.intersection(true_free_cells)
    )

    coverage = (
        visited_free_cells / len(true_free_cells) * 100
        if true_free_cells
        else 0
    )

    safety_checks = status["safety_checks"]
    safety_overrides = status["safety_overrides"]

    override_percentage = (
        safety_overrides / safety_checks * 100
        if safety_checks > 0
        else 0
    )

    return {
        "algorithm": algorithm.upper(),
        "seed": seed,
        "coverage": coverage,
        "visited_cells": visited_cells,
        "visited_free_cells": visited_free_cells,
        "true_free_cells": len(true_free_cells),
        "collisions": env.collisions,
        "collision_episode": int(env.collisions > 0),
        "frontiers_reached": status["frontiers_reached"],
        "steps": env.steps,
        "replans": status["replans"],
        "safety_checks": safety_checks,
        "safety_overrides": safety_overrides,
        "override_percentage": override_percentage,
        "safety_stops": status["safety_stops"],
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


def print_summary(all_results):

    metrics = [
        ("coverage", "True free-space coverage (%)"),
        ("visited_cells", "All visited grid cells"),
        ("visited_free_cells", "Visited traversable cells"),
        ("true_free_cells", "Total traversable cells"),
        ("collisions", "Collisions"),
        ("collision_episode", "Episodes with collisions"),
        ("frontiers_reached", "Frontiers reached"),
        ("steps", "Steps"),
        ("replans", "Replans"),
        ("safety_checks", "Safety checks"),
        ("safety_overrides", "Safety overrides"),
        ("override_percentage", "Override percentage (%)"),
        ("safety_stops", "Safety stops"),
    ]

    print("\n========== FINAL 30-SEED COMPARISON ==========")

    for algorithm in ALGORITHMS:

        results = [
            result
            for result in all_results
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


def save_results(all_results):

    os.makedirs("results", exist_ok=True)

    output_file = "results/fuzzy_planner_comparison.csv"

    fieldnames = [
        "algorithm",
        "seed",
        "coverage",
        "visited_cells",
        "visited_free_cells",
        "true_free_cells",
        "collisions",
        "collision_episode",
        "frontiers_reached",
        "steps",
        "replans",
        "safety_checks",
        "safety_overrides",
        "override_percentage",
        "safety_stops",
    ]

    with open(output_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nIndividual results saved to:", output_file)


def main():

    all_results = []

    for algorithm in ALGORITHMS:

        print(
            "\nRunning {} experiments...".format(
                algorithm.upper()
            )
        )

        for seed in range(
            START_SEED,
            START_SEED + NUM_SEEDS
        ):

            result = run_episode(algorithm, seed)

            all_results.append(result)

            print(
                "{} | Seed: {} | Coverage: {:.2f}% | "
                "Visited: {} | Free visited: {} | "
                "Collisions: {} | Frontiers: {} | "
                "Overrides: {}/{} | Stops: {}".format(
                    algorithm.upper(),
                    seed,
                    result["coverage"],
                    result["visited_cells"],
                    result["visited_free_cells"],
                    result["collisions"],
                    result["frontiers_reached"],
                    result["safety_overrides"],
                    result["safety_checks"],
                    result["safety_stops"]
                )
            )

        print(
            "{} experiments completed.".format(
                algorithm.upper()
            )
        )

    print_summary(all_results)
    save_results(all_results)

    print("\n========== EVALUATION COMPLETE ==========")
    print("Total episodes:", len(all_results))


if __name__ == "__main__":
    main()
