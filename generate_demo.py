import math
import os
import shutil
import subprocess
import tempfile

import pygame

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ROBOT_START,
    ROBOT_RADIUS,
)

from lidar.simulator import IndoorEnvironment
from lidar.ray_casting import perform_scan
from simulation.renderer import Renderer


FPS = 10
OUTPUT_DIR = "assets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lidar-demo.gif")


def can_move(environment, position):
    x, y = position

    checks = [
        (x, y),
        (x - ROBOT_RADIUS, y),
        (x + ROBOT_RADIUS, y),
        (x, y - ROBOT_RADIUS),
        (x, y + ROBOT_RADIUS),
        (x - 9, y - 9),
        (x + 9, y - 9),
        (x - 9, y + 9),
        (x + 9, y + 9),
    ]

    return all(
        not environment.is_obstacle(px, py)
        for px, py in checks
    )


def draw_demo_label(screen):
    panel = pygame.Rect(15, 10, 475, 125)

    pygame.draw.rect(
        screen, (245, 247, 250), panel, border_radius=8
    )
    pygame.draw.rect(
        screen, (45, 48, 58), panel, 2, border_radius=8
    )

    font = pygame.font.SysFont("Arial", 18, bold=True)
    small_font = pygame.font.SysFont("Arial", 15)

    lines = [
        (font, "SCRIPTED LiDAR SIMULATION DEMO"),
        (small_font, "180-ray scan | Maximum range: 350 px"),
        (small_font, "Green lines: LiDAR obstacle detections"),
        (small_font, "Red points: Detected obstacle locations"),
    ]

    y = 19

    for current_font, message in lines:
        text = current_font.render(message, True, (25, 30, 40))
        screen.blit(text, (28, y))
        y += 27


def main():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is not installed or not in PATH.")

    pygame.init()

    screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

    environment = IndoorEnvironment(
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    renderer = Renderer(screen)

    # A closed, scripted route through clear areas of the room.
    waypoints = [
        (150, 150),
        (150, 280),
        (280, 280),
        (540, 280),
        (720, 280),
        (720, 320),
        (720, 520),
        (550, 520),
        (270, 520),
        (270, 300),
        (100, 300),
        (100, 150),
        (150, 150),
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x, y = ROBOT_START
    frame_number = 0

    try:
        with tempfile.TemporaryDirectory(
            prefix="lidar_demo_"
        ) as temp_dir:

            for target in waypoints[1:]:
                start_x, start_y = x, y
                target_x, target_y = target

                distance = math.hypot(
                    target_x - start_x,
                    target_y - start_y
                )

                frame_count = max(
                    2, round(distance / 14)
                )

                heading = math.degrees(
                    math.atan2(
                        target_y - start_y,
                        target_x - start_x
                    )
                )

                for step in range(1, frame_count + 1):
                    progress = step / frame_count

                    current_x = (
                        start_x +
                        (target_x - start_x) * progress
                    )

                    current_y = (
                        start_y +
                        (target_y - start_y) * progress
                    )

                    position = (
                        int(round(current_x)),
                        int(round(current_y))
                    )

                    if not can_move(environment, position):
                        raise RuntimeError(
                            f"Demo route blocked at {position}"
                        )

                    scan_results = perform_scan(
                        environment,
                        position,
                        heading
                    )

                    renderer.draw_environment(environment)
                    renderer.draw_lidar(position, scan_results)
                    renderer.draw_robot(position, heading)

                    draw_demo_label(screen)

                    frame_path = os.path.join(
                        temp_dir,
                        f"frame_{frame_number:04d}.png"
                    )

                    pygame.image.save(screen, frame_path)
                    frame_number += 1

                # Update the starting position for the next waypoint.
                x, y = target_x, target_y

            print(f"Generated {frame_number} frames.")

            frame_pattern = os.path.join(
                temp_dir, "frame_%04d.png"
            )

            palette_path = os.path.join(
                temp_dir, "palette.png"
            )

            # Generate a GIF colour palette.
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(FPS),
                    "-i", frame_pattern,
                    "-vf",
                    "fps=10,scale=800:-1:flags=lanczos,"
                    "palettegen=max_colors=128",
                    palette_path,
                ],
                check=True,
            )

            # Convert frames into an optimized looping GIF.
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(FPS),
                    "-i", frame_pattern,
                    "-i", palette_path,
                    "-lavfi",
                    "fps=10,scale=800:-1:flags=lanczos[x];"
                    "[x][1:v]paletteuse=dither=bayer:"
                    "bayer_scale=5",
                    "-loop", "0",
                    OUTPUT_FILE,
                ],
                check=True,
            )

        print("\nDemo created successfully!")
        print(f"Location: {OUTPUT_FILE}")

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
