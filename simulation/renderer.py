
import math
import pygame

from config import ROBOT_RADIUS


class Renderer:

    def __init__(self, screen):

        self.screen = screen

        self.background = (238, 240, 244)
        self.wall_color = (45, 48, 58)
        self.obstacle_color = (135, 145, 160)
        self.robot_color = (30, 110, 220)
        self.lidar_color = (40, 170, 120)

        self.font = pygame.font.SysFont("Arial", 19)

    def draw_environment(self, environment):

        self.screen.fill(self.background)

        # Room boundary
        pygame.draw.rect(
            self.screen,
            self.wall_color,
            (
                0,
                0,
                environment.width,
                environment.height
            ),
            environment.wall_thickness
        )

        # Furniture
        for obstacle in environment.obstacles:

            pygame.draw.rect(
                self.screen,
                self.obstacle_color,
                obstacle,
                border_radius=5
            )

            pygame.draw.rect(
                self.screen,
                (90, 100, 115),
                obstacle,
                width=2,
                border_radius=5
            )

    def draw_lidar(self, position, scan_results):

        for measurement in scan_results:

            endpoint = measurement["endpoint"]

            # Show detected obstacle points
            if measurement["hit"]:

                pygame.draw.line(
                    self.screen,
                    self.lidar_color,
                    position,
                    endpoint,
                    1
                )

                pygame.draw.circle(
                    self.screen,
                    (220, 60, 60),
                    endpoint,
                    3
                )

            else:

                pygame.draw.line(
                    self.screen,
                    (175, 195, 185),
                    position,
                    endpoint,
                    1
                )

    def draw_robot(self, position, heading):

        pygame.draw.circle(
            self.screen,
            self.robot_color,
            position,
            ROBOT_RADIUS
        )

        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            position,
            ROBOT_RADIUS,
            2
        )

        radians = math.radians(heading)

        front_x = position[0] + math.cos(radians) * 23
        front_y = position[1] + math.sin(radians) * 23

        pygame.draw.line(
            self.screen,
            (255, 220, 50),
            position,
            (front_x, front_y),
            4
        )

    def draw_information(self, scan_results, heading):

        detected = sum(
            1 for result in scan_results
            if result["hit"]
        )

        lines = [
            "AUTONOMOUS INDOOR EXPLORATION ROBOT",
            f"LiDAR rays: {len(scan_results)}",
            f"Obstacle detections: {detected}",
            f"Robot heading: {heading:.1f} degrees",
            "WASD: Move | Q/E: Rotate | ESC: Exit"
        ]

        y = 12

        for line in lines:

            text = self.font.render(
                line,
                True,
                (25, 30, 40)
            )

            self.screen.blit(text, (35, y))

            y += 25
