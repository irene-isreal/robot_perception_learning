
import os
import csv
import math
import random

import numpy as np


# LiDAR configuration
NUM_RAYS = 72
MAX_RANGE = 250
ANGLE_INCREMENT = 360 / NUM_RAYS

CLASSES = ["person", "box", "shelf", "free_space"]


def ray_circle_intersection(direction, center, radius):

    dx, dy = direction
    cx, cy = center

    projection = cx * dx + cy * dy

    perpendicular_sq = (
        cx * cx + cy * cy - projection * projection
    )

    if perpendicular_sq > radius * radius:
        return None

    offset = math.sqrt(
        radius * radius - perpendicular_sq
    )

    distance = projection - offset

    if distance <= 0:
        distance = projection + offset

    if distance <= 0:
        return None

    return distance


def ray_rectangle_intersection(
    direction,
    center,
    width,
    height,
    orientation
):

    dx, dy = direction
    cx, cy = center

    angle = math.radians(orientation)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Transform ray into the rectangle's local coordinate system
    ox = -cx * cos_a - cy * sin_a
    oy = cx * sin_a - cy * cos_a

    local_dx = dx * cos_a + dy * sin_a
    local_dy = -dx * sin_a + dy * cos_a

    half_width = width / 2
    half_height = height / 2

    t_min = -float("inf")
    t_max = float("inf")

    # Intersection with rectangle's horizontal boundaries
    if abs(local_dx) < 1e-9:

        if abs(ox) > half_width:
            return None

    else:

        tx1 = (-half_width - ox) / local_dx
        tx2 = (half_width - ox) / local_dx

        t_min = max(t_min, min(tx1, tx2))
        t_max = min(t_max, max(tx1, tx2))

    # Intersection with rectangle's vertical boundaries
    if abs(local_dy) < 1e-9:

        if abs(oy) > half_height:
            return None

    else:

        ty1 = (-half_height - oy) / local_dy
        ty2 = (half_height - oy) / local_dy

        t_min = max(t_min, min(ty1, ty2))
        t_max = min(t_max, max(ty1, ty2))

    if t_max < t_min or t_max < 0:
        return None

    distance = t_min if t_min > 0 else t_max

    return distance if distance > 0 else None


def generate_scan(object_type, noise_std=1.5):

    # Empty scene
    if object_type == "free_space":

        return np.full(NUM_RAYS, MAX_RANGE)

    # Random object position relative to sensor
    distance = random.uniform(70, 200)

    bearing = math.radians(
        random.uniform(0, 360)
    )

    center = (
        distance * math.cos(bearing),
        distance * math.sin(bearing)
    )

    orientation = random.uniform(0, 180)

    if object_type == "person":

        radius = random.uniform(12, 22)

    elif object_type == "box":

        width = random.uniform(20, 55)
        height = random.uniform(20, 55)

    elif object_type == "shelf":

        width = random.uniform(70, 130)
        height = random.uniform(15, 30)

    else:

        raise ValueError(
            f"Unknown object type: {object_type}"
        )

    measurements = []

    for i in range(NUM_RAYS):

        angle = math.radians(
            i * ANGLE_INCREMENT
        )

        direction = (
            math.cos(angle),
            math.sin(angle)
        )

        if object_type == "person":

            hit_distance = ray_circle_intersection(
                direction,
                center,
                radius
            )

        else:

            hit_distance = ray_rectangle_intersection(
                direction,
                center,
                width,
                height,
                orientation
            )

        if hit_distance is None or hit_distance > MAX_RANGE:

            measured_distance = MAX_RANGE

        else:

            # Add simulated sensor noise
            measured_distance = (
                hit_distance
                + np.random.normal(0, noise_std)
            )

            measured_distance = np.clip(
                measured_distance,
                0,
                MAX_RANGE
            )

        measurements.append(measured_distance)

    return np.array(measurements)


def generate_dataset(
    samples_per_class=1000,
    output_path="results/lidar_dataset.csv"
):

    random.seed(42)
    np.random.seed(42)

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    feature_names = [
        f"ray_{i:02d}"
        for i in range(NUM_RAYS)
    ]

    fieldnames = feature_names + ["label"]

    total_samples = 0

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for object_type in CLASSES:

            print(
                f"Generating {samples_per_class} "
                f"samples for: {object_type}"
            )

            for _ in range(samples_per_class):

                scan = generate_scan(object_type)

                row = {
                    feature_names[i]: round(float(scan[i]), 3)
                    for i in range(NUM_RAYS)
                }

                row["label"] = object_type

                writer.writerow(row)

                total_samples += 1

    print("\nDataset generation complete!")
    print(f"Total samples: {total_samples}")
    print(f"Features per sample: {NUM_RAYS}")
    print(f"Classes: {CLASSES}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    generate_dataset()
