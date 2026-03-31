import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cfar import CFAR_PFA, cfar2d_polar_ca, suggest_default_params
from utils.data_loading import _normalize_polar_layout, cartesian_to_polar_image, polar_to_cartesian_image


def extract_shoreline(
    img,
    min_cluster_area_m2=64,
    morph_open_k=3,
    morph_close_k=7,
    grid_size=4096,
    cart_threshold=0.0,
    polar_threshold=0.5,
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
    cart_mask_float, x, y = polar_to_cartesian_image(detections.astype(np.float32), grid_size=grid_size)
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
    polar_mask_normalized = cartesian_to_polar_image(cart_mask.astype(np.float32), polar_shape=(n_range, n_azimuth))
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

    return shoreline_points, cart_mask, polar_mask


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
    shoreline_points, cart_mask, polar_mask = extract_shoreline(img)

    print(f"Detected {len(shoreline_points)} shoreline points.")

    # Convert the full image to Cartesian for the background
    cart_img, x_coords, y_coords = polar_to_cartesian_image(img, grid_size=1024)

    # Transform the detected shoreline polar coordinates to Cartesian directly
    cart_shoreline_x = []
    cart_shoreline_y = []
    if shoreline_points:
        azs, ranges = zip(*shoreline_points)
        # polar_to_cartesian_points takes (ranges, angles)
        pts = polar_to_cartesian_points(np.array(ranges), np.array(azs))
        cart_shoreline_x = pts[:, 0]
        cart_shoreline_y = pts[:, 1]

    print("Clustering shoreline points...")
    cluster_labels = cluster_shoreline_dbscan(cart_shoreline_x, cart_shoreline_y)

    from utils.visualisation import plot_shoreline_extraction

    plot_shoreline_extraction(img, shoreline_points, cart_mask, polar_mask, cluster_labels)
