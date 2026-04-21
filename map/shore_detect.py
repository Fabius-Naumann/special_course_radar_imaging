import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cfar import CFAR_PFA, cfar2d_polar_ca, suggest_default_params  # noqa: E402
from utils.data_loading import (  # noqa: E402
    _normalize_polar_layout,
    cartesian_to_polar_image,
    polar_to_cartesian_image,
    polar_to_cartesian_points,
)


def _build_shoreline_metadata(
    shoreline_points,
    cart_points,
    cluster_labels,
    min_segment_points=5,
    min_segment_length_m=20.0,
):
    """Compute per-point segment ids, arc lengths, and quality weights."""
    num_points = len(shoreline_points)
    if num_points == 0:
        return {
            "cart_points": np.empty((0, 2), dtype=float),
            "cluster_labels": np.empty((0,), dtype=int),
            "segment_ids": np.empty((0,), dtype=int),
            "arc_lengths_m": np.empty((0,), dtype=float),
            "segment_lengths_m": np.empty((0,), dtype=float),
            "quality_weights": np.empty((0,), dtype=float),
            "valid_mask": np.empty((0,), dtype=bool),
        }

    cluster_labels = np.asarray(cluster_labels, dtype=int)
    cart_points = np.asarray(cart_points, dtype=float)
    azimuth_indices = np.asarray([point[0] for point in shoreline_points], dtype=int)

    segment_ids = np.full(num_points, -1, dtype=int)
    arc_lengths = np.zeros(num_points, dtype=float)
    segment_lengths = np.zeros(num_points, dtype=float)
    quality_weights = np.full(num_points, 0.05, dtype=float)

    next_segment_id = 0
    unique_labels = [label for label in sorted(set(cluster_labels.tolist())) if label >= 0]
    for label_value in unique_labels:
        member_indices = np.where(cluster_labels == label_value)[0]
        if member_indices.size == 0:
            continue

        ordered = member_indices[np.argsort(azimuth_indices[member_indices])]
        if ordered.size > 1:
            diffs = np.linalg.norm(np.diff(cart_points[ordered], axis=0), axis=1)
            segment_length_m = float(np.sum(diffs))
            local_spacing = np.full(ordered.size, np.inf, dtype=float)
            local_spacing[0] = diffs[0]
            local_spacing[-1] = diffs[-1]
            if ordered.size > 2:
                local_spacing[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
            local_density = 1.0 / np.maximum(local_spacing, 1.0)
            local_density /= max(float(np.max(local_density)), 1.0)
            arc = np.zeros(ordered.size, dtype=float)
            arc[1:] = np.cumsum(diffs)
        else:
            segment_length_m = 0.0
            local_density = np.ones(ordered.size, dtype=float)
            arc = np.zeros(ordered.size, dtype=float)

        size_quality = min(1.0, ordered.size / max(float(min_segment_points * 2), 1.0))
        length_quality = min(1.0, segment_length_m / max(float(min_segment_length_m * 2.0), 1.0))
        cluster_quality = 0.5 * local_density + 0.25 * size_quality + 0.25 * length_quality

        if ordered.size >= int(min_segment_points) and segment_length_m >= float(min_segment_length_m):
            segment_ids[ordered] = next_segment_id
            next_segment_id += 1
            quality_weights[ordered] = np.clip(cluster_quality, 0.2, 1.0)
        else:
            quality_weights[ordered] = np.clip(0.15 * cluster_quality, 0.01, 0.15)

        arc_lengths[ordered] = arc
        segment_lengths[ordered] = segment_length_m

    return {
        "cart_points": cart_points,
        "cluster_labels": cluster_labels,
        "segment_ids": segment_ids,
        "arc_lengths_m": arc_lengths,
        "segment_lengths_m": segment_lengths,
        "quality_weights": quality_weights,
        "valid_mask": segment_ids >= 0,
    }


def extract_shoreline(
    img,
    min_cluster_area_m2=64,
    morph_open_k=3,
    morph_close_k=7,
    grid_size=4096,
    cart_threshold=0.0,
    polar_threshold=0.5,
    clockwise_azimuth=False,
    cluster_min_samples=5,
    cluster_cut_distance_m=10.0,
    min_segment_points=5,
    min_segment_length_m=20.0,
    return_metadata=False,
):
    """
    Extract shoreline points from the radar image.

    Parameters:
    - img: 2D array of the radar image in polar coordinates.
    - min_cluster_area_m2: Minimum area (in square meters) for a cluster of detections to be kept.
    - morph_open_k: Structure size for morphological opening (removes isolated noise).
    - morph_close_k: Structure size for morphological closing (fills gaps).

    Returns:
    - shoreline_points: List of tuples (azimuth_idx, range_idx) for detected shoreline points.
    """
    # 1. CFAR Detection
    detections, threshold, noise, valid = cfar2d_polar_ca(
        img,
        **suggest_default_params(),
        normalize_azimuth=True,
        range_pad_mode="edge",
        pfa=CFAR_PFA,
        min_noise_floor_factor=1.3,
    )

    # 2. Polar to Cartesian mask conversion
    # polar_to_cartesian_image manages the transpose automatically via _normalize_polar_layout
    cart_mask_float, x, y = polar_to_cartesian_image(
        detections.astype(np.float32),
        grid_size=grid_size,
        clockwise_azimuth=clockwise_azimuth,
    )
    cart_mask = cart_mask_float > cart_threshold

    # 3. Morphological filtering in Cartesian
    if morph_close_k > 0:
        struct_close = np.ones((morph_close_k, morph_close_k), dtype=bool)
        cart_mask = binary_closing(cart_mask, structure=struct_close, iterations=2)

    if morph_open_k > 0:
        struct_open = np.ones((morph_open_k, morph_open_k), dtype=bool)
        cart_mask = binary_opening(cart_mask, structure=struct_open, iterations=1)

    if morph_close_k > 0:
        struct_close = np.ones((morph_close_k, morph_close_k), dtype=bool)
        cart_mask = binary_closing(cart_mask, structure=struct_close, iterations=1)

    # Area filtering
    if min_cluster_area_m2 > 0:
        labeled_mask, n_features = label(cart_mask)
        clean_mask = np.zeros_like(cart_mask)
        # Convert m^2 to pixel count based on the Cartesian grid resolution
        min_cluster_area_px = max(1, round(min_cluster_area_m2 / (x[-1] - x[0]) * grid_size))
        for i in range(1, n_features + 1):
            if (labeled_mask == i).sum() >= min_cluster_area_px:
                clean_mask[labeled_mask == i] = True
        cart_mask = clean_mask

    # 4. Cartesian to Polar mapping
    n_range, n_azimuth = _normalize_polar_layout(img).shape
    polar_mask_normalized = cartesian_to_polar_image(
        cart_mask.astype(np.float32),
        polar_shape=(n_range, n_azimuth),
        clockwise_azimuth=clockwise_azimuth,
    )
    polar_mask_normalized = polar_mask_normalized > polar_threshold

    # Match original shape
    if img.shape[0] == n_azimuth:  # meaning original was (azimuth, range)
        polar_mask = polar_mask_normalized.T
        az_axis = 0
        r_axis = 1
    else:
        polar_mask = polar_mask_normalized
        az_axis = 1
        r_axis = 0

    # 5. Extract shoreline points (first true detection in each azimuth)
    shoreline_points = []

    for az in range(n_azimuth):
        # Slice to get all range bins for this azimuth
        range_slice = polar_mask[az, :] if az_axis == 0 else polar_mask[:, az]

        range_indices = np.where(range_slice)[0]
        if len(range_indices) > 0:
            first_range = range_indices[0]  # lowest range index
            # Append (az_idx, range_idx)
            shoreline_points.append((az, first_range))

    if not return_metadata:
        return shoreline_points, cart_mask, polar_mask, cart_mask_float

    if shoreline_points:
        azimuth_indices, range_indices = zip(*shoreline_points, strict=False)
        cart_points = polar_to_cartesian_points(
            np.asarray(range_indices, dtype=float),
            np.asarray(azimuth_indices, dtype=float),
            clockwise_azimuth=clockwise_azimuth,
        )
        cluster_labels = cluster_shoreline_dbscan(
            cart_points[:, 0],
            cart_points[:, 1],
            min_samples=cluster_min_samples,
            cut_distance=cluster_cut_distance_m,
        )
    else:
        cart_points = np.empty((0, 2), dtype=float)
        cluster_labels = np.empty((0,), dtype=int)

    metadata = _build_shoreline_metadata(
        shoreline_points,
        cart_points,
        cluster_labels,
        min_segment_points=min_segment_points,
        min_segment_length_m=min_segment_length_m,
    )
    return shoreline_points, cart_mask, polar_mask, cart_mask_float, metadata


def cluster_shoreline_dbscan(cart_x, cart_y, min_samples=5, cut_distance=10, **kwargs):
    """
    Cluster shoreline points using DBSCAN.

    Parameters:
    - cart_x: Array of x coordinates
    - cart_y: Array of y coordinates
    - min_samples: Number of samples in a neighborhood for a point to be considered as a core point
    - cut_distance: Neighborhood radius in Cartesian coordinates (meters)

    Returns:
    - labels: Cluster labels for each point. Noisy samples are given the label -1.
    """
    if len(cart_x) == 0:
        return np.array([])

    points = np.column_stack((cart_x, cart_y))
    clusterer = DBSCAN(eps=cut_distance, min_samples=min_samples, **kwargs)
    return clusterer.fit_predict(points)


if __name__ == "__main__":
    from utils.data_loading import load_radar_images, polar_to_cartesian_points

    print("Loading radar images...")
    filenames, images = load_radar_images(num_images=1)
    img = images[0]

    print("Extracting shoreline...")
    shoreline_points, cart_mask, polar_mask, _ = extract_shoreline(img)

    print(f"Detected {len(shoreline_points)} shoreline points.")

    # Convert the full image to Cartesian for the background
    cart_img, x_coords, y_coords = polar_to_cartesian_image(img, grid_size=1024)

    # Transform the detected shoreline polar coordinates to Cartesian directly
    cart_shoreline_x = []
    cart_shoreline_y = []
    if shoreline_points:
        azs, ranges = zip(*shoreline_points, strict=False)
        # polar_to_cartesian_points takes (ranges, angles)
        pts = polar_to_cartesian_points(np.array(ranges), np.array(azs))
        cart_shoreline_x = pts[:, 0]
        cart_shoreline_y = pts[:, 1]

    print("Clustering shoreline points...")
    cluster_labels = cluster_shoreline_dbscan(cart_shoreline_x, cart_shoreline_y)

    from utils.visualisation import plot_shoreline_extraction

    plot_shoreline_extraction(img, shoreline_points, cart_mask, polar_mask, cluster_labels)
