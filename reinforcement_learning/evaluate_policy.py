
import math
import random
import os

import pandas as pd

from reinforcement_learning.environment import (
    ExplorationEnvironment
)

from reinforcement_learning.agent import (
    QLearningAgent
)


def evaluate():

    random.seed(123)

    environment = ExplorationEnvironment()
    agent = QLearningAgent()

    agent.load(
        "models/frontier_q_table.pkl"
    )

    agent.epsilon = 0

    episodes = 30

    results = []

    print("\n======================================")
    print(" FRONTIER Q-LEARNING EVALUATION")
    print("======================================")

    print(f"Episodes: {episodes}")
    print(
        f"Traversable grid cells: "
        f"{environment.true_free_cells}"
    )

    for episode in range(episodes):

        state = environment.reset()

        start_x = environment.x
        start_y = environment.y

        positions = [(start_x, start_y)]

        cell_sequence = [
            environment.get_grid_cell(
                start_x, start_y
            )
        ]

        done = False

        while not done:

            action = agent.choose_action(state)

            next_state, reward, done, info = (
                environment.step(action)
            )

            state = next_state

            positions.append((
                environment.x,
                environment.y
            ))

            current_cell = environment.get_grid_cell(
                environment.x,
                environment.y
            )

            if current_cell != cell_sequence[-1]:
                cell_sequence.append(current_cell)

        unique_cells = len(set(cell_sequence))

        coverage = (
            unique_cells /
            environment.true_free_cells
        ) * 100

        transitions = len(cell_sequence)

        revisits = transitions - unique_cells

        if transitions > 1:

            revisit_rate = (
                revisits / (transitions - 1)
            ) * 100

        else:

            revisit_rate = 0

        max_distance = max(
            math.hypot(
                x - start_x,
                y - start_y
            )
            for x, y in positions
        )

        known_cells = int(
            environment.occupancy.astype(bool).sum()
        )

        map_coverage = (
            known_cells /
            (environment.grid_rows * environment.grid_cols)
        ) * 100

        collision = info["collisions"] > 0

        row = {

            "Episode": episode + 1,
            "Coverage_Percent": coverage,
            "Unique_Cells": unique_cells,
            "Revisit_Rate": revisit_rate,
            "Collisions": int(collision),
            "Steps": info["steps"],
            "Goals_Reached": info["goals_reached"],
            "Known_Map_Cells": known_cells,
            "Map_Coverage_Percent": map_coverage,
            "Max_Distance": max_distance

        }

        results.append(row)

        print(
            f"Episode {episode + 1:02d}: "
            f"Coverage={coverage:.2f}% | "
            f"Unique={unique_cells} | "
            f"Revisits={revisit_rate:.1f}% | "
            f"Goals={info['goals_reached']} | "
            f"Collision={collision} | "
            f"Steps={info['steps']} | "
            f"Map known={map_coverage:.1f}%"
        )

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    df.to_csv(
        "results/frontier_evaluation.csv",
        index=False
    )

    print("\n======================================")
    print("         FINAL EVALUATION")
    print("======================================")

    print(
        f"Average room coverage: "
        f"{df['Coverage_Percent'].mean():.2f}%"
    )

    print(
        f"Best room coverage: "
        f"{df['Coverage_Percent'].max():.2f}%"
    )

    print(
        f"Average unique cells: "
        f"{df['Unique_Cells'].mean():.2f}"
    )

    print(
        f"Average revisit rate: "
        f"{df['Revisit_Rate'].mean():.2f}%"
    )

    print(
        f"Collision rate: "
        f"{df['Collisions'].mean() * 100:.2f}%"
    )

    print(
        f"Average frontier goals reached: "
        f"{df['Goals_Reached'].mean():.2f}"
    )

    print(
        f"Average mapped area: "
        f"{df['Map_Coverage_Percent'].mean():.2f}%"
    )

    print(
        f"Average maximum distance: "
        f"{df['Max_Distance'].mean():.2f}px"
    )

    print("\nSaved: results/frontier_evaluation.csv")
    print("======================================")

if __name__ == "__main__":
    evaluate()
