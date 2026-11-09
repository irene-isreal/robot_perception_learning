
import math

from config import (
    LIDAR_RAYS,
    LIDAR_MAX_RANGE,
    LIDAR_STEP_SIZE
)


def cast_ray(environment, origin, angle):

    x, y = origin

    radians = math.radians(angle)

    direction_x = math.cos(radians)
    direction_y = math.sin(radians)

    distance = 0

    while distance < LIDAR_MAX_RANGE:

        distance += LIDAR_STEP_SIZE

        ray_x = x + direction_x * distance
        ray_y = y + direction_y * distance

        if environment.is_obstacle(ray_x, ray_y):

            return (
                (int(ray_x), int(ray_y)),
                distance,
                True
            )

    # No obstacle detected within maximum range
    end_x = x + direction_x * LIDAR_MAX_RANGE
    end_y = y + direction_y * LIDAR_MAX_RANGE

    return (
        (int(end_x), int(end_y)),
        LIDAR_MAX_RANGE,
        False
    )


def perform_scan(environment, position, heading=0):

    scan_results = []

    angle_increment = 360 / LIDAR_RAYS

    for i in range(LIDAR_RAYS):

        angle = heading + i * angle_increment

        endpoint, distance, hit = cast_ray(
            environment,
            position,
            angle
        )

        scan_results.append({
            "angle": angle % 360,
            "distance": distance,
            "endpoint": endpoint,
            "hit": hit
        })

    return scan_results
