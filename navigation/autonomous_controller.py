
import math

from navigation.grid_planner import plan_frontier_path
from safety.fuzzy_controller import FuzzySafetyController


class AutonomousController:

    def __init__(self, algorithm="astar"):
        algorithm = algorithm.lower()

        if algorithm not in ("astar", "bfs"):
            raise ValueError(
                "Algorithm must be 'astar' or 'bfs'."
            )

        self.algorithm = algorithm
        self.reset()

    def reset(self):
        self.path = []
        self.goal = None
        self.next_index = 1

        self.replans = 0
        self.actions_taken = 0
        self.frontiers_reached = 0

        self.safety_controller = FuzzySafetyController(
            max_range=180
        )

        self.safety_checks = 0
        self.safety_overrides = 0
        self.safety_stops = 0

        # Temporarily rejected movement cells only.
        # Frontier goals are no longer suppressed.
        self.temporary_blocked_cells = {}
        self.block_memory_duration = 30

        # Obstacle escape state.
        self.escape_turn_action = None
        self.escape_turn_steps = 0
        self.max_escape_turn_steps = 4
        self.escape_clearance = 85
        self.escape_switched_direction = False

        # Brief forward movement after escaping.
        self.escape_forward_steps = 0

    def get_robot_cell(self, env):
        gx, gy = env.world_to_grid(env.x, env.y)
        return gy, gx

    def get_sector_distance(
        self,
        env,
        center_angle,
        offsets
    ):
        readings = [
            env.measure_distance(
                (center_angle + offset) % 360
            )
            for offset in offsets
        ]

        return min(readings)

    def get_lidar_distances(self, env):
        front_distance = self.get_sector_distance(
            env,
            env.heading,
            (-25, -15, 0, 15, 25)
        )

        left_distance = self.get_sector_distance(
            env,
            env.heading - 90,
            (-15, 0, 15)
        )

        right_distance = self.get_sector_distance(
            env,
            env.heading + 90,
            (-15, 0, 15)
        )

        return (
            front_distance,
            left_distance,
            right_distance
        )

    def remember_blocked_cell(self, cell):
        if cell is None:
            return

        expiry = (
            self.actions_taken
            + self.block_memory_duration
        )

        self.temporary_blocked_cells[cell] = expiry

    def get_planning_grid(self, env, start):
        # Work on a copy; never modify the real occupancy map.
        planning_grid = env.occupancy.copy()

        # Remove expired movement blocks.
        expired_cells = [
            cell
            for cell, expiry
            in self.temporary_blocked_cells.items()
            if self.actions_taken >= expiry
        ]

        for cell in expired_cells:
            del self.temporary_blocked_cells[cell]

        rows, cols = planning_grid.shape

        # Temporarily avoid only cells rejected by safety.
        for row, col in self.temporary_blocked_cells:
            if (
                0 <= row < rows
                and 0 <= col < cols
                and (row, col) != start
            ):
                planning_grid[row, col] = -1

        return planning_grid

    def plan_route(self, env):
        start = self.get_robot_cell(env)

        planning_grid = self.get_planning_grid(
            env,
            start
        )

        # Do not suppress frontier goals.
        # A rejected movement cell does not mean the
        # overall frontier is unreachable.
        result = plan_frontier_path(
            planning_grid,
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

        if (
            self.goal is not None
            and current_cell == self.goal
        ):
            self.frontiers_reached += 1

            self.path = []
            self.goal = None
            self.next_index = 1

        if not self.path:
            result = self.plan_route(env)

            if not result["success"]:
                return None

        elif self.next_index < len(self.path):
            expected_previous_cell = (
                self.path[self.next_index - 1]
            )

            expected_next_cell = (
                self.path[self.next_index]
            )

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

        target_row, target_col = (
            self.path[self.next_index]
        )

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

            if front_distance >= self.escape_clearance:

                self.escape_turn_action = None
                self.escape_turn_steps = 0
                self.escape_switched_direction = False

                self.escape_forward_steps = 2
                self.clear_route()

                self.actions_taken += 1
                return 0

            if (
                self.escape_turn_steps
                < self.max_escape_turn_steps
            ):
                self.escape_turn_steps += 1
                self.actions_taken += 1

                return self.escape_turn_action

            if not self.escape_switched_direction:

                self.escape_turn_action = (
                    3
                    if self.escape_turn_action == 2
                    else 2
                )

                self.escape_turn_steps = 0
                self.escape_switched_direction = True

                self.actions_taken += 1

                return self.escape_turn_action

            self.safety_stops += 1

            self.escape_turn_action = None
            self.escape_turn_steps = 0
            self.escape_switched_direction = False
            self.escape_forward_steps = 0

            self.clear_route()

            return None

        # --------------------------------------------------
        # 2. Brief forward movement after a successful escape.
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

            self.escape_forward_steps = 0

            left_distance = self.get_sector_distance(
                env,
                env.heading - 90,
                (-15, 0, 15)
            )

            right_distance = self.get_sector_distance(
                env,
                env.heading + 90,
                (-15, 0, 15)
            )

            self.escape_turn_action = (
                2
                if left_distance >= right_distance
                else 3
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

            safety_result = (
                self.safety_controller.get_action(
                    front_distance,
                    left_distance,
                    right_distance
                )
            )

            safety_action = safety_result["action"]

            if safety_action != 0:

                final_action = safety_action
                self.safety_overrides += 1

                # Remember only the immediate cell that
                # the robot was attempting to enter.
                if self.next_index < len(self.path):

                    blocked_cell = (
                        self.path[self.next_index]
                    )

                    self.remember_blocked_cell(
                        blocked_cell
                    )

                # Keep the frontier available for a new
                # route after the obstacle escape.
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
            "path_length": max(
                0,
                len(self.path) - 1
            ),
            "replans": self.replans,
            "actions_taken": self.actions_taken,
            "frontiers_reached": self.frontiers_reached,
            "safety_checks": self.safety_checks,
            "safety_overrides": self.safety_overrides,
            "safety_stops": self.safety_stops,
            "escape_turn_steps": self.escape_turn_steps,
            "escape_forward_steps": self.escape_forward_steps,
            "escape_active": (
                self.escape_turn_action is not None
            ),
            "temporarily_blocked_cells": len(
                self.temporary_blocked_cells
            )
        }
