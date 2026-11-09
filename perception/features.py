
import numpy as np


def extract_features(scan, max_range=250):

    scan = np.asarray(scan, dtype=float)

    # Normalise raw distances
    normalized = np.clip(
        scan / max_range,
        0,
        1
    )

    features = []

    # 1. Global statistical features
    features.extend([
        np.min(normalized),
        np.max(normalized),
        np.mean(normalized),
        np.std(normalized),
        np.median(normalized),
        np.percentile(normalized, 25),
        np.percentile(normalized, 75)
    ])

    # 2. Obstacle detection features
    hit_mask = normalized < 0.98

    features.extend([
        np.mean(hit_mask),
        np.sum(hit_mask)
    ])

    # Count separate obstacle regions in the circular scan
    transitions = np.sum(
        hit_mask != np.roll(hit_mask, 1)
    )

    features.append(transitions / 2)

    # 3. Geometric variation between neighbouring rays
    differences = np.diff(
        np.append(normalized, normalized[0])
    )

    features.extend([
        np.mean(np.abs(differences)),
        np.std(differences),
        np.max(np.abs(differences))
    ])

    # 4. Divide scan into 12 angular sectors
    sectors = np.array_split(normalized, 12)

    for sector in sectors:

        features.extend([
            np.min(sector),
            np.mean(sector),
            np.std(sector)
        ])

    # 5. Include all normalised ray measurements
    features.extend(normalized.tolist())

    return np.asarray(features, dtype=float)


def transform_dataset(X):

    return np.array([
        extract_features(scan)
        for scan in X
    ])
