
import math
import pygame

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    FPS,
    WINDOW_TITLE,
    ROBOT_START,
    ROBOT_SPEED,
    ROTATION_SPEED,
    ROBOT_RADIUS
)

from lidar.simulator import IndoorEnvironment
from lidar.ray_casting import perform_scan
from simulation.renderer import Renderer


def can_move(environment, position):

    x, y = position

    # Check the robot's centre and surrounding body
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
        not environment.is_obstacle(px, py)
        for px, py in checks
    )


def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )

    pygame.display.set_caption(WINDOW_TITLE)

    clock = pygame.time.Clock()

    environment = IndoorEnvironment(
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    renderer = Renderer(screen)

    x, y = ROBOT_START
    heading = 0

    running = True

    while running:

        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()

        # Robot rotation
        if keys[pygame.K_q]:
            heading -= ROTATION_SPEED

        if keys[pygame.K_e]:
            heading += ROTATION_SPEED

        heading %= 360

        # Forward/backward movement
        movement = 0

        if keys[pygame.K_w]:
            movement += ROBOT_SPEED

        if keys[pygame.K_s]:
            movement -= ROBOT_SPEED

        # Sideways movement
        side_movement = 0

        if keys[pygame.K_a]:
            side_movement -= ROBOT_SPEED

        if keys[pygame.K_d]:
            side_movement += ROBOT_SPEED

        angle = math.radians(heading)

        dx = (
            math.cos(angle) * movement
            - math.sin(angle) * side_movement
        )

        dy = (
            math.sin(angle) * movement
            + math.cos(angle) * side_movement
        )

        new_position = (x + dx, y + dy)

        if can_move(environment, new_position):

            x, y = new_position

        robot_position = (int(x), int(y))

        # Perform a fresh LiDAR scan
        scan_results = perform_scan(
            environment,
            robot_position,
            heading
        )

        # Render everything
        renderer.draw_environment(environment)

        renderer.draw_lidar(
            robot_position,
            scan_results
        )

        renderer.draw_robot(
            robot_position,
            heading
        )

        renderer.draw_information(
            scan_results,
            heading
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
