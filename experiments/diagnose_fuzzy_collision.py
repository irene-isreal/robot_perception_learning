
import random

from reinforcement_learning.environment import ExplorationEnvironment
from navigation.autonomous_controller import AutonomousController
from navigation.grid_planner import get_frontier_cells


SEED = 69
ALGORITHM = "bfs"
MAX_STEPS = 250


def main():

    # Match the random seed and initialization order
    # used in evaluate_fuzzy_planners.py.
    random.seed(SEED)

    env = ExplorationEnvironment()
    controller = AutonomousController(
        algorithm=ALGORITHM
    )

    print("\n========== EXPLORATION DIAGNOSTIC ==========")
    print("Seed:", SEED)
    print("Algorithm:", ALGORITHM.upper())
    print("Starting position:", (env.x, env.y))
    print("Starting heading:", env.heading)
    print("Maximum steps:", MAX_STEPS)
    print()

    action_names = {
        0: "FORWARD",
        1: "BACKWARD",
        2: "LEFT",
        3: "RIGHT"
    }

    previous_goal = None
    previous_replans = 0
    previous_frontiers = 0

    for step in range(MAX_STEPS):

        before_position = (env.x, env.y)
        before_heading = env.heading

        front, left, right = (
            controller.get_lidar_distances(env)
        )

        action = controller.get_action(env)

        status = controller.get_status()

        current_goal = status["goal"]
        frontier_count = len(
            get_frontier_cells(env.occupancy)
        )

        if action is None:

            print(
                f"Step {step + 1:03d} | "
                "NO ACTION / NO REACHABLE FRONTIER"
            )

            print("Controller status:", status)
            break

        _, reward, done, info = env.step(action)

        goal_changed = current_goal != previous_goal

        replan_changed = (
            status["replans"] != previous_replans
        )

        frontier_reached = (
            status["frontiers_reached"] != previous_frontiers
        )

        # Print every step so turns and movement can be traced.
        print(
            f"Step {step + 1:03d} | "
            f"Pos: ({before_position[0]:.1f}, "
            f"{before_position[1]:.1f}) | "
            f"Heading: {before_heading:.0f} | "
            f"F/L/R: {front}/{left}/{right} | "
            f"Action: {action_names[action]} | "
            f"Goal: {current_goal} | "
            f"Frontiers: {frontier_count} | "
            f"Visited: {len(env.visited_cells)} | "
            f"Replans: {status['replans']} | "
            f"Overrides: {status['safety_overrides']} | "
            f"Reward: {reward:.2f}"
        )

        if (
            goal_changed
            or replan_changed
            or frontier_reached
        ):
            print(
                "    EVENT | "
                f"Goal changed: {goal_changed} | "
                f"Replanned: {replan_changed} | "
                f"Frontier reached: {frontier_reached} | "
                f"New goal: {current_goal}"
            )

        previous_goal = current_goal
        previous_replans = status["replans"]
        previous_frontiers = status["frontiers_reached"]

        if env.collisions > 0:
            print("\n!!! COLLISION DETECTED !!!")
            break

        if done:
            print("\nEpisode ended.")
            break

    final_status = controller.get_status()

    print("\n========== FINAL DIAGNOSTIC SUMMARY ==========")
    print("Final position:", (env.x, env.y))
    print("Final heading:", env.heading)
    print("Steps:", env.steps)
    print("Visited cells:", len(env.visited_cells))
    print("Remaining frontiers:", len(
        get_frontier_cells(env.occupancy)
    ))
    print("Collisions:", env.collisions)
    print("Controller status:", final_status)

    print("\n========== DIAGNOSTIC COMPLETE ==========")


if __name__ == "__main__":
    main()
