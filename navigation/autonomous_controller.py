
import math

from navigation.grid_planner import plan_frontier_path
from safety.fuzzy_controller import FuzzySafetyController


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

        self.safety_controller = FuzzySafetyController(max_range=180)

        self.safety_checks = 0
        self.safety_overrides = 0
        self.safety_stops = 0

        # Obstacle escape state
        self.escape_turn_action = None
        self.escape_turn_steps = 0
        self.max_escape_turn_steps = 4
        self.escape_clearance = 85
        self.escape_switched_direction = False

        # Move forward briefly after finding a clear direction.
        self.escape_forward_steps = 0

    def get_robot_cell(self, env):
        gx, gy = env.world_to_grid(env.x, env.y)
        return gy, gx

    def get_sector_distance(self, env, center_angle, offsets):
        readings = [
            env.measure_distance((center_angle + offset) % 360)
            for offset in offsets
        ]
        return min(readings)

    def get_lidar_distances(self, env):
        front_distance = self.get_sector_distance(
            env, env.heading, (-25, -15, 0, 15, 25)
        )

        left_distance = self.get_sector_distance(
            env, env.heading - 90, (-15, 0, 15)
        )

        right_distance = self.get_sector_distance(
            env, env.heading + 90, (-15, 0, 15)
        )

        return front_distance, left_distance, right_distance

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

    def get_planned_action(self, env):
        current_cell = self.get_robot_cell(env)

        if self.goal is not None and current_cell == self.goal:
            self.frontiers_reached += 1
            self.path = []
            self.goal = None
            self.next_index = 1

        if not self.path:
            result = self.plan_route(env)

            if not result["success"]:
                return None

        elif self.next_index < len(self.path):
            expected_previous_cell = self.path[self.next_index - 1]
            expected_next_cell = self.path[self.next_index]

            if current_cell == expected_next_cell:
                self.next_index += 1

            elif current_cell != expected_previous_cell:
                result = self.plan_route(env)

                if not result["success"]:
                    return None

        if self.next_index >= len(self.path):
            result = self.plan_route(env)

            if not result["success"]:
                return None

        if self.next_index >= len(self.path):
            return None

        target_row, target_col = self.path[self.next_index]

        target_x, target_y = env.grid_to_world(
            target_col, target_row
        )

        current_row, current_col = current_cell

        current_x, current_y = env.grid_to_world(
            current_col, current_row
        )

        dx = target_x - current_x
        dy = target_y - current_y

        desired_heading = math.degrees(
            math.atan2(dy, dx)
        ) % 360

        difference = (
            desired_heading - env.heading + 180
        ) % 360 - 180

        if difference > 22.5:
            return 3

        if difference < -22.5:
            return 2

        return 0

    def clear_route(self):
        self.path = []
        self.goal = None
        self.next_index = 1

    def get_action(self, env):

        # --------------------------------------------------
        # 1. Continue obstacle escape behaviour.
        # --------------------------------------------------

        if self.escape_turn_action is not None:

            front_distance = self.get_sector_distance(
                env,
                env.heading,
                (-25, -15, 0, 15, 25)
            )

            self.safety_checks += 1

            # A clear direction has been found.
            if front_distance >= self.escape_clearance:
                self.escape_turn_action = None
                self.escape_turn_steps = 0
                self.escape_switched_direction = False

                # Move away from the obstacle before replanning.
                self.escape_forward_steps = 2

                self.clear_route()

                self.actions_taken += 1
                return 0

            # Continue turning in the same direction.
            if self.escape_turn_steps < self.max_escape_turn_steps:
                self.escape_turn_steps += 1
                self.actions_taken += 1
                return self.escape_turn_action

            # First direction did not work. Try the other side.
            if not self.escape_switched_direction:
                self.escape_turn_action = (
                    3 if self.escape_turn_action == 2 else 2
                )

                self.escape_turn_steps = 0
                self.escape_switched_direction = True

                self.actions_taken += 1
                return self.escape_turn_action

            # Neither direction produced enough clearance.
            self.safety_stops += 1

            self.escape_turn_action = None
            self.escape_turn_steps = 0
            self.escape_switched_direction = False
            self.escape_forward_steps = 0

            self.clear_route()

            return None

        # --------------------------------------------------
        # 2. Briefly move forward after a successful escape.
        # --------------------------------------------------

        if self.escape_forward_steps > 0:

            front_distance = self.get_sector_distance(
                env,
                env.heading,
                (-25, -15, 0, 15, 25)
            )

            self.safety_checks += 1

            if front_distance >= self.escape_clearance:
                self.escape_forward_steps -= 1
                self.actions_taken += 1
                return 0

            # The path became blocked again.
            self.escape_forward_steps = 0

            self.escape_turn_action = (
                2 if self.get_sector_distance(
                    env, env.heading - 90, (-15, 0, 15)
                ) >= self.get_sector_distance(
                    env, env.heading + 90, (-15, 0, 15)
                ) else 3
            )

            self.escape_turn_steps = 0
            self.escape_switched_direction = False

            self.safety_overrides += 1
            self.clear_route()

            self.actions_taken += 1
            return self.escape_turn_action

        # --------------------------------------------------
        # 3. Obtain the planned navigation action.
        # --------------------------------------------------

        planned_action = self.get_planned_action(env)

        if planned_action is None:
            return None

        final_action = planned_action

        # --------------------------------------------------
        # 4. Apply fuzzy safety before moving forward.
        # --------------------------------------------------

        if planned_action == 0:

            (
                front_distance,
                left_distance,
                right_distance
            ) = self.get_lidar_distances(env)

            self.safety_checks += 1

            safety_result = self.safety_controller.get_action(
                front_distance,
                left_distance,
                right_distance
            )

            safety_action = safety_result["action"]

            if safety_action != 0:
                final_action = safety_action
                self.safety_overrides += 1

                self.escape_turn_action = safety_action
                self.escape_turn_steps = 0
                self.escape_switched_direction = False

                self.clear_route()

        self.actions_taken += 1

        return final_action

    def get_status(self):
        return {
            "algorithm": self.algorithm.upper(),
            "goal": self.goal,
            "path_length": max(0, len(self.path) - 1),
            "replans": self.replans,
            "actions_taken": self.actions_taken,
            "frontiers_reached": self.frontiers_reached,
            "safety_checks": self.safety_checks,
            "safety_overrides": self.safety_overrides,
            "safety_stops": self.safety_stops,
            "escape_turn_steps": self.escape_turn_steps,
            "escape_forward_steps": self.escape_forward_steps,
            "escape_active": self.escape_turn_action is not None
        }
