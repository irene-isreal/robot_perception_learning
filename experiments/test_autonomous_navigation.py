
import random

from reinforcement_learning.environment import ExplorationEnvironment
from navigation.autonomous_controller import AutonomousController


def run_episode(algorithm, seed=42):

    random.seed(seed)

    env = ExplorationEnvironment()

    controller = AutonomousController(
        algorithm=algorithm
    )

    total_reward = 0
    done = False

    while not done:

        action = controller.get_action(env)

        if action is None:
            print("No reachable frontier found.")
            break

        _, reward, done, info = env.step(action)

        total_reward += reward

    status = controller.get_status()

    visited_cells = len(env.visited_cells)

    total_cells = env.grid_rows * env.grid_cols

    coverage = (visited_cells / total_cells) * 100

    print("\n========== {} RESULTS ==========".format(
        algorithm.upper()
    ))

    print("Total reward:", round(total_reward, 2))
    print("Visited cells:", visited_cells)
    print("Room coverage: {:.2f}%".format(coverage))
    print("Controller frontiers reached:",
          status["frontiers_reached"])
    print("Collisions:", env.collisions)
    print("Steps:", env.steps)
    print("Replans:", status["replans"])
    print("Actions taken:", status["actions_taken"])
    print("Final goal:", status["goal"])

    return {
        "algorithm": algorithm.upper(),
        "coverage": coverage,
        "visited_cells": visited_cells,
        "collisions": env.collisions,
        "frontiers_reached": status["frontiers_reached"],
        "steps": env.steps,
        "replans": status["replans"],
    }


def main():

    results = []

    for algorithm in ("bfs", "astar"):
        result = run_episode(algorithm, seed=42)
        results.append(result)

    print("\n========== BFS vs A* COMPARISON ==========")

    for result in results:
        print(
            "{} | Coverage: {:.2f}% | "
            "Collisions: {} | Steps: {} | "
            "Frontiers reached: {}".format(
                result["algorithm"],
                result["coverage"],
                result["collisions"],
                result["steps"],
                result["frontiers_reached"]
            )
        )


if __name__ == "__main__":
    main()
