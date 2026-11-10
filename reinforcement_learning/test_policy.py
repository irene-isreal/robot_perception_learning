import numpy as np

from reinforcement_learning.environment import (
    ExplorationEnvironment
)

from reinforcement_learning.agent import (
    QLearningAgent
)


environment = ExplorationEnvironment()
agent = QLearningAgent()

agent.load("models/q_learning_table.npy")
agent.epsilon = 0

state = environment.reset()

action_names = [
    "Forward",
    "Backward",
    "Left",
    "Right"
]

print("\nGREEDY POLICY TEST")
print("-" * 55)

for step in range(40):

    action = int(
        np.argmax(agent.q_table[state])
    )

    q_values = agent.q_table[state]

    print(
        f"Step {step + 1:02d} | "
        f"State: {state:4d} | "
        f"Action: {action_names[action]:9s} | "
        f"Q-values: {np.round(q_values, 2)}"
    )

    next_state, reward, done, info = (
        environment.step(action)
    )

    state = next_state

    if done:
        print("\nEpisode ended.")
        break

print("\nFinal position:", environment.x, environment.y)
print("Visited cells:", len(environment.visited_cells))
print("Collisions:", environment.collisions)