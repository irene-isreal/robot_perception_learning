
import math

from navigation.grid_planner import plan_frontier_path


class AutonomousController:

    def __init__(self, algorithm="astar"):
        algorithm = algorithm.lower()

        if algorithm not in ("astar", "bfs"):
            raise ValueError("Algorithm must be 'astar' or 'bfs'.")

        self.algorithm = algorithm
        self.reset()

    def reset(self):
        self.path = []
        self.goal = None
        self.next_index = 1

        self.replans = 0
        self.actions_taken = 0
        self.frontiers_reached = 0

    def get_robot_cell(self, env):
        gx, gy = env.world_to_grid(env.x, env.y)

        # The planner uses (row, column), not (x, y).
        return gy, gx

    def plan_route(self, env):
        start = self.get_robot_cell(env)

        result = plan_frontier_path(
            env.occupancy,
            start,
            algorithm=self.algorithm
        )

        self.path = result["path"] or []
        self.goal = result["goal"]
        self.next_index = 1

        self.replans += 1

        return result

    def get_action(self, env):
        current_cell = self.get_robot_cell(env)

        # Check whether the robot has reached its current frontier.
        if self.goal is not None and current_cell == self.goal:
            self.frontiers_reached += 1

            self.path = []
            self.goal = None
            self.next_index = 1

        # If there is no active path, select a new frontier.
        if not self.path:
            result = self.plan_route(env)

            if not result["success"]:
                return None

        # Check whether the robot has entered the next planned cell.
        elif self.next_index < len(self.path):

            expected_previous_cell = self.path[self.next_index - 1]
            expected_next_cell = self.path[self.next_index]

            if current_cell == expected_next_cell:
                self.next_index += 1

            elif current_cell != expected_previous_cell:
                # Robot deviated from its planned route.
                result = self.plan_route(env)

                if not result["success"]:
                    return None

        # If the route is exhausted, plan another one.
        if self.next_index >= len(self.path):
            result = self.plan_route(env)

            if not result["success"]:
                return None

        if self.next_index >= len(self.path):
            return None

        target_row, target_col = self.path[self.next_index]

        target_x, target_y = env.grid_to_world(
            target_col,
            target_row
        )

        current_row, current_col = current_cell

        current_x, current_y = env.grid_to_world(
            current_col,
            current_row
        )

        dx = target_x - current_x
        dy = target_y - current_y

        desired_heading = math.degrees(
            math.atan2(dy, dx)
        ) % 360

        difference = (
            desired_heading - env.heading + 180
        ) % 360 - 180

        # The environment turns by 45 degrees per action.
        if difference > 22.5:
            action = 3  # Turn right

        elif difference < -22.5:
            action = 2  # Turn left

        else:
            action = 0  # Move forward

        self.actions_taken += 1

        return action

    def get_status(self):
        return {
            "algorithm": self.algorithm.upper(),
            "goal": self.goal,
            "path_length": max(0, len(self.path) - 1),
            "replans": self.replans,
            "actions_taken": self.actions_taken,
            "frontiers_reached": self.frontiers_reached,
        }
