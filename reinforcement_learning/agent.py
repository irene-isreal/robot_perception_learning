import pickle
import random
import numpy as np


class QLearningAgent:

    def __init__(
        self,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):

        self.action_size = action_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Sparse Q-table: only encountered states are stored
        self.q_table = {}

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(
                self.action_size
            )

        return self.q_table[state]

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        # Exploitation with random tie-breaking
        q_values = self.get_q_values(state)

        best_value = np.max(q_values)

        best_actions = np.flatnonzero(
            np.isclose(q_values, best_value)
        )

        return int(
            random.choice(best_actions.tolist())
        )

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        current_q_values = self.get_q_values(state)

        current_q = current_q_values[action]

        if done:

            target = reward

        else:

            next_q_values = self.get_q_values(
                next_state
            )

            target = (
                reward
                + self.discount_factor
                * np.max(next_q_values)
            )

        current_q_values[action] += (
            self.learning_rate
            * (target - current_q)
        )

    def decay_exploration(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save(self, path):

        with open(path, "wb") as file:

            pickle.dump(
                self.q_table,
                file
            )

    def load(self, path):

        with open(path, "rb") as file:

            self.q_table = pickle.load(file)