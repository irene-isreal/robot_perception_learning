
import random
import numpy as np

from reinforcement_learning.environment import ExplorationEnvironment
from navigation.autonomous_controller import AutonomousController


def main():

    seed = 44

    random.seed(seed)
    np.random.seed(seed)

    env = ExplorationEnvironment()
    controller = AutonomousController(algorithm="bfs")

    print("\n========== COLLISION DIAGNOSTIC ==========\n")

    action_names = {
        0: "FORWARD",
        1: "BACKWARD",
        2: "LEFT",
        3: "RIGHT"
    }

    for step in range(150):

        before_position = (env.x, env.y)
        before_heading = env.heading

        front, left, right = (
            controller.get_lidar_distances(env)
        )

        action = controller.get_action(env)

        if action is None:
            print("No reachable frontier.")
            break

        _, reward, done, info = env.step(action)

        print(
            f"Step {step + 1:03d} | "
            f"Position: ({before_position[0]:.1f}, "
            f"{before_position[1]:.1f}) | "
            f"Heading: {before_heading:.0f} | "
            f"Front sector: {front} | "
            f"Left sector: {left} | "
            f"Right sector: {right} | "
            f"Action: {action_names[action]} | "
            f"Reward: {reward:.2f}"
        )

        if env.collisions > 0:

            print("\n!!! COLLISION DETECTED !!!")
            print(f"Step: {step + 1}")
            print(f"Position before collision: {before_position}")
            print(f"Heading before collision: {before_heading}")
            print(
                f"Sector readings: "
                f"F={front}, L={left}, R={right}"
            )
            print(f"Action: {action_names[action]}")
            print(
                f"Position after step: "
                f"({env.x}, {env.y})"
            )
            print(
                f"Safety status: {controller.get_status()}"
            )

            break

        if done:
            print("\nEpisode ended without collision.")
            break

    print("\n========== DIAGNOSTIC COMPLETE ==========")


if __name__ == "__main__":
    main()
