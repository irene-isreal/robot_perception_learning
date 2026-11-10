
import pygame

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    FPS
)

from reinforcement_learning.environment import (
    ExplorationEnvironment
)

from reinforcement_learning.agent import (
    QLearningAgent
)

from lidar.ray_casting import perform_scan
from simulation.renderer import Renderer


def draw_occupancy_map(screen, environment):

    overlay = pygame.Surface(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.SRCALPHA
    )

    cell_size = environment.cell_size

    for gy in range(environment.grid_rows):

        for gx in range(environment.grid_cols):

            value = environment.occupancy[gy, gx]

            rect = pygame.Rect(
                gx * cell_size,
                gy * cell_size,
                cell_size,
                cell_size
            )

            if value == 1:

                pygame.draw.rect(
                    overlay,
                    (40, 180, 80, 65),
                    rect
                )

            elif value == -1:

                pygame.draw.rect(
                    overlay,
                    (220, 60, 60, 100),
                    rect
                )

    screen.blit(overlay, (0, 0))


def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )

    pygame.display.set_caption(
        "Frontier-Based Autonomous Exploration"
    )

    clock = pygame.time.Clock()

    environment = ExplorationEnvironment()
    agent = QLearningAgent()

    agent.load(
        "models/frontier_q_table.pkl"
    )

    agent.epsilon = 0

    renderer = Renderer(screen)

    state = environment.reset()

    running = True
    paused = False
    done = False

    total_reward = 0
    path = []

    frame_counter = 0
    decision_interval = 4

    font = pygame.font.SysFont("Arial", 18)

    action_names = [
        "Forward",
        "Backward",
        "Turn Left",
        "Turn Right"
    ]

    current_action = "Starting"

    while running:

        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:

                    state = environment.reset()

                    total_reward = 0
                    path = []

                    done = False
                    paused = False

                    frame_counter = 0
                    current_action = "Restarted"

        if not paused and not done:

            frame_counter += 1

            if frame_counter >= decision_interval:

                frame_counter = 0

                action = agent.choose_action(state)

                current_action = action_names[action]

                next_state, reward, done, info = (
                    environment.step(action)
                )

                state = next_state
                total_reward += reward

                path.append((
                    int(environment.x),
                    int(environment.y)
                ))

        renderer.draw_environment(
            environment.environment
        )

        draw_occupancy_map(
            screen,
            environment
        )

        if environment.goal_position is not None:

            pygame.draw.circle(
                screen,
                (30, 100, 255),
                (
                    int(environment.goal_position[0]),
                    int(environment.goal_position[1])
                ),
                9
            )

        if len(path) > 1:

            pygame.draw.lines(
                screen,
                (255, 150, 40),
                False,
                path,
                2
            )

        robot_position = (
            int(environment.x),
            int(environment.y)
        )

        scan_results = perform_scan(
            environment.environment,
            robot_position,
            environment.heading
        )

        renderer.draw_lidar(
            robot_position,
            scan_results
        )

        renderer.draw_robot(
            robot_position,
            environment.heading
        )

        renderer.draw_information(
            scan_results,
            environment.heading
        )

        information = [

            f"Action: {current_action}",
            f"Visited cells: {len(environment.visited_cells)}",
            f"Goals reached: {environment.goals_reached}",
            f"Known map cells: {environment.occupancy.astype(bool).sum()}",
            f"Total reward: {total_reward:.2f}",
            f"Steps: {environment.steps}",
            "Green: Free | Red: Occupied | Black: Unknown",
            "Blue: Current frontier target",
            "SPACE: Pause | R: Restart | ESC: Exit"

        ]

        y = 115

        for line in information:

            text = font.render(
                line,
                True,
                (20, 30, 45)
            )

            screen.blit(
                text,
                (30, y)
            )

            y += 25

        if paused:

            message = font.render(
                "PAUSED",
                True,
                (200, 80, 30)
            )

            screen.blit(
                message,
                (WINDOW_WIDTH - 130, 20)
            )

        if done:

            message = font.render(
                "EPISODE FINISHED - Press R to restart",
                True,
                (200, 50, 50)
            )

            screen.blit(
                message,
                (WINDOW_WIDTH // 2 - 230, WINDOW_HEIGHT - 40)
            )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
