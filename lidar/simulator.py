
import pygame


class IndoorEnvironment:

    def __init__(self, width, height):

        self.width = width
        self.height = height

        self.wall_thickness = 20

        # Rectangular furniture and obstacles
        self.obstacles = [

            # Sofa
            pygame.Rect(300, 100, 180, 65),

            # Table
            pygame.Rect(600, 130, 100, 100),

            # Cabinet
            pygame.Rect(150, 350, 100, 160),

            # Desk
            pygame.Rect(400, 400, 180, 65),

            # Storage unit
            pygame.Rect(750, 350, 100, 150),

            # Small obstacles
            pygame.Rect(650, 550, 65, 65),
            pygame.Rect(350, 570, 75, 45)
        ]

    def is_obstacle(self, x, y):

        # Room boundaries
        if x < self.wall_thickness:
            return True

        if y < self.wall_thickness:
            return True

        if x >= self.width - self.wall_thickness:
            return True

        if y >= self.height - self.wall_thickness:
            return True

        # Furniture
        for obstacle in self.obstacles:

            if obstacle.collidepoint(x, y):
                return True

        return False
