
import math
import random
from collections import deque

import numpy as np

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ROBOT_START,
    ROBOT_RADIUS
)

from lidar.simulator import IndoorEnvironment


class ExplorationEnvironment:

    def __init__(self):

        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT

        self.environment = IndoorEnvironment(
            self.width,
            self.height
        )

        self.cell_size = 30

        self.grid_cols = math.ceil(
            self.width / self.cell_size
        )

        self.grid_rows = math.ceil(
            self.height / self.cell_size
        )

        # Occupancy values:
        # -1 = occupied
        #  0 = unknown
        #  1 = free

        self.max_range = 180
        self.map_ray_step = 10
        self.map_ray_count = 16

        self.policy_ray_count = 8
        self.policy_ray_step = 4

        self.move_distance = 8
        self.turn_angle = 45
        self.max_steps = 250

        self.actions = [
            "forward",
            "backward",
            "left",
            "right"
        ]

        self.true_free_cells = self.count_true_free_cells()

        self.reset()

    # --------------------------------------------------
    # GRID UTILITIES
    # --------------------------------------------------

    def world_to_grid(self, x, y):

        gx = int(x // self.cell_size)
        gy = int(y // self.cell_size)

        gx = max(0, min(gx, self.grid_cols - 1))
        gy = max(0, min(gy, self.grid_rows - 1))

        return gx, gy

    def grid_to_world(self, gx, gy):

        return (
            gx * self.cell_size + self.cell_size / 2,
            gy * self.cell_size + self.cell_size / 2
        )

    def get_grid_cell(self, x, y):

        return self.world_to_grid(x, y)

    def count_true_free_cells(self):

        count = 0

        for gy in range(self.grid_rows):
            for gx in range(self.grid_cols):

                x, y = self.grid_to_world(gx, gy)

                if self.is_valid_position(x, y):
                    count += 1

        return max(count, 1)

    # --------------------------------------------------
    # COLLISION CHECKING
    # --------------------------------------------------

    def is_valid_position(self, x, y):

        checks = [
            (x, y),
            (x - ROBOT_RADIUS, y),
            (x + ROBOT_RADIUS, y),
            (x, y - ROBOT_RADIUS),
            (x, y + ROBOT_RADIUS),
            (x - 9, y - 9),
            (x + 9, y - 9),
            (x - 9, y + 9),
            (x + 9, y + 9)
        ]

        return all(
            not self.environment.is_obstacle(px, py)
            for px, py in checks
        )

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self):

        self.occupancy = np.zeros(
            (self.grid_rows, self.grid_cols),
            dtype=np.int8
        )

        self.visited_cells = set()
        self.state_visit_counts = {}

        self.goals_reached = 0
        self.steps = 0
        self.collisions = 0

        self.goal_cell = None
        self.goal_position = None

        # Find a random valid starting position.
        for _ in range(200):

            x = random.randint(60, self.width - 60)
            y = random.randint(60, self.height - 60)

            if self.is_valid_position(x, y):
                self.x = float(x)
                self.y = float(y)
                break

        else:

            self.x = float(ROBOT_START[0])
            self.y = float(ROBOT_START[1])

        self.heading = random.choice(
            [0, 45, 90, 135, 180, 225, 270, 315]
        )

        self.visited_cells.add(
            self.world_to_grid(self.x, self.y)
        )

        # Build the initial local map.
        self.update_occupancy_map()

        self.select_frontier_goal()

        initial_state = self.get_state()
        self.state_visit_counts[initial_state] = 1

        return initial_state

    # --------------------------------------------------
    # LIDAR MAPPING
    # --------------------------------------------------

    def measure_distance(self, angle):

        radians = math.radians(angle)

        dx = math.cos(radians)
        dy = math.sin(radians)

        distance = 0

        while distance < self.max_range:

            distance += self.policy_ray_step

            px = self.x + dx * distance
            py = self.y + dy * distance

            if self.environment.is_obstacle(px, py):
                return distance

        return self.max_range

    def update_occupancy_map(self):

        robot_cell = self.world_to_grid(
            self.x, self.y
        )

        self.occupancy[
            robot_cell[1], robot_cell[0]
        ] = 1

        for ray in range(self.map_ray_count):

            angle = (
                self.heading
                + ray * 360 / self.map_ray_count
            )

            radians = math.radians(angle)

            dx = math.cos(radians)
            dy = math.sin(radians)

            hit_obstacle = False

            for distance in range(
                self.map_ray_step,
                self.max_range + 1,
                self.map_ray_step
            ):

                px = self.x + dx * distance
                py = self.y + dy * distance

                gx, gy = self.world_to_grid(px, py)

                if self.environment.is_obstacle(px, py):

                    self.occupancy[gy, gx] = -1
                    hit_obstacle = True
                    break

                # Mark the traversed cell as free.
                if self.occupancy[gy, gx] != -1:
                    self.occupancy[gy, gx] = 1

            # No obstacle was detected within the range.
            # Traversed cells remain marked as free.
            if not hit_obstacle:
                pass

    # --------------------------------------------------
    # FRONTIER DETECTION
    # --------------------------------------------------

    def get_neighbours(self, gx, gy):

        return [
            (gx + 1, gy),
            (gx - 1, gy),
            (gx, gy + 1),
            (gx, gy - 1)
        ]

    def is_frontier(self, gx, gy):

        if self.occupancy[gy, gx] != 1:
            return False

        for nx, ny in self.get_neighbours(gx, gy):

            if (
                0 <= nx < self.grid_cols
                and 0 <= ny < self.grid_rows
            ):

                if self.occupancy[ny, nx] == 0:
                    return True

        return False

    def select_frontier_goal(self):

        start = self.world_to_grid(
            self.x, self.y
        )

        queue = deque([(start, 0)])
        visited = {start}

        candidates = []

        while queue:

            (gx, gy), distance = queue.popleft()

            # Do not select the robot's current cell.
            if (gx, gy) != start:

                if self.is_frontier(gx, gy):

                    unknown_neighbours = sum(
                        1
                        for nx, ny in self.get_neighbours(gx, gy)
                        if (
                            0 <= nx < self.grid_cols
                            and 0 <= ny < self.grid_rows
                            and self.occupancy[ny, nx] == 0
                        )
                    )

                    score = (
                        distance
                        - 0.75 * unknown_neighbours
                    )

                    candidates.append(
                        (score, gx, gy)
                    )

            for nx, ny in self.get_neighbours(gx, gy):

                if not (
                    0 <= nx < self.grid_cols
                    and 0 <= ny < self.grid_rows
                ):
                    continue

                if (nx, ny) in visited:
                    continue

                if self.occupancy[ny, nx] != 1:
                    continue

                visited.add((nx, ny))

                queue.append(
                    ((nx, ny), distance + 1)
                )

        if not candidates:

            self.goal_cell = None
            self.goal_position = None
            return

        candidates.sort(
            key=lambda item: item[0]
        )

        _, gx, gy = candidates[0]

        self.goal_cell = (gx, gy)

        self.goal_position = self.grid_to_world(
            gx, gy
        )

    # --------------------------------------------------
    # Q-LEARNING STATE
    # --------------------------------------------------

    def get_state(self):

        readings = []

        for i in range(self.policy_ray_count):

            angle = (
                self.heading
                + i * 360 / self.policy_ray_count
            )

            distance = self.measure_distance(angle)

            if distance <= 35:
                level = 0
            elif distance <= 80:
                level = 1
            else:
                level = 2

            readings.append(level)

        lidar_code = 0

        for reading in readings:
            lidar_code = lidar_code * 3 + reading

        if self.goal_position is None:

            target_sector = 0
            distance_bin = 0
            goal_present = 0

        else:

            goal_x, goal_y = self.goal_position

            dx = goal_x - self.x
            dy = goal_y - self.y

            distance = math.hypot(dx, dy)

            bearing = math.degrees(
                math.atan2(dy, dx)
            ) % 360

            relative_angle = (
                bearing - self.heading
            ) % 360

            target_sector = int(
                (relative_angle + 22.5) // 45
            ) % 8

            if distance <= 60:
                distance_bin = 0
            elif distance <= 120:
                distance_bin = 1
            else:
                distance_bin = 2

            goal_present = 1

        return (
            lidar_code,
            target_sector,
            distance_bin,
            goal_present
        )

    # --------------------------------------------------
    # ENVIRONMENT STEP
    # --------------------------------------------------

    def step(self, action):

        self.steps += 1

        reward = -0.1
        done = False

        old_position = (self.x, self.y)

        old_goal_distance = None

        if self.goal_position is not None:

            old_goal_distance = math.hypot(
                self.goal_position[0] - self.x,
                self.goal_position[1] - self.y
            )

        if action == 0:

            distance = self.move_distance

        elif action == 1:

            distance = -self.move_distance

        elif action == 2:

            self.heading -= self.turn_angle
            distance = 0

        elif action == 3:

            self.heading += self.turn_angle
            distance = 0

        else:

            raise ValueError("Invalid action")

        self.heading %= 360

        if distance != 0:

            radians = math.radians(self.heading)

            new_x = self.x + math.cos(radians) * distance
            new_y = self.y + math.sin(radians) * distance

            if self.is_valid_position(new_x, new_y):

                self.x = new_x
                self.y = new_y

            else:

                reward -= 12
                self.collisions += 1
                done = True

        current_cell = self.world_to_grid(
            self.x, self.y
        )

        old_cell = self.world_to_grid(
            old_position[0],
            old_position[1]
        )

        if current_cell != old_cell:

            if current_cell not in self.visited_cells:

                self.visited_cells.add(current_cell)
                reward += 2.0

            else:

                reward -= 0.2

        # Update the robot's map after movement.
        self.update_occupancy_map()

        # Check whether the current frontier goal is reached.
        if (
            self.goal_position is not None
            and not done
        ):

            new_goal_distance = math.hypot(
                self.goal_position[0] - self.x,
                self.goal_position[1] - self.y
            )

            # Reward progress towards the selected goal.
            progress = (
                old_goal_distance - new_goal_distance
            )

            reward += 0.04 * progress

            if new_goal_distance <= self.cell_size * 0.7:

                reward += 10
                self.goals_reached += 1

                self.select_frontier_goal()

        # Replan if the previous target is no longer free.
        if self.goal_cell is not None:

            gx, gy = self.goal_cell

            if self.occupancy[gy, gx] != 1:
                self.select_frontier_goal()

        elif not done:

            self.select_frontier_goal()

        next_state = self.get_state()

        state_visits = self.state_visit_counts.get(
            next_state, 0
        )

        if state_visits > 0:
            reward -= 0.05

        self.state_visit_counts[next_state] = (
            state_visits + 1
        )

        if self.steps >= self.max_steps:
            done = True

        known_cells = int(
            np.count_nonzero(self.occupancy)
        )

        info = {

            "visited_cells": len(self.visited_cells),
            "collisions": self.collisions,
            "steps": self.steps,
            "goals_reached": self.goals_reached,
            "known_cells": known_cells,
            "coverage": (
                len(self.visited_cells)
                / self.true_free_cells
            ) * 100

        }

        return next_state, reward, done, info
