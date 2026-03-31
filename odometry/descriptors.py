import numpy as np
import cv2
from scipy.spatial import cKDTree

import sys
from pathlib import Path

# Add the project root folder to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils.data_loading import polar_to_cartesian_points

def compute_descriptors(img, keypoints, alpha=18, rho=10, max_radius=50):
    """
    Compute rotation-invariant descriptors for radar keypoints.
    
    Parameters:
    -----------
    img : ndarray
        The radar image (polar coordinates: angles x ranges)
    keypoints : ndarray
        Array of keypoints with shape (N, 2) where each row is (angle_idx, range_idx)
    alpha : int
        Number of angular slices for angular histogram
    rho : int
        Number of annuli for radial histogram
    max_radius : int
        Maximum radius around keypoint to consider for descriptor
    
    Returns:
    --------
    descriptors : ndarray
        Array of shape (N, alpha+rho) containing descriptors for each keypoint
    """
    descriptors = []
    num_angles, num_ranges = img.shape
    
    for kp in keypoints:
        a_kp, r_kp = int(kp[0]), int(kp[1])
        
        # Initialize histograms
        angular_hist = np.zeros(alpha)
        radial_hist = np.zeros(rho)
        
        # Define region around keypoint
        angle_start = max(0, a_kp - max_radius)
        angle_end = min(num_angles, a_kp + max_radius + 1)
        range_start = max(0, r_kp - max_radius)
        range_end = min(num_ranges, r_kp + max_radius + 1)
        
        # Iterate over neighborhood
        for a in range(angle_start, angle_end):
            for r in range(range_start, range_end):
                # Compute relative position
                da = a - a_kp
                dr = r - r_kp
                
                # Skip if outside max_radius
                dist = np.sqrt(da**2 + dr**2)
                if dist > max_radius or dist == 0:
                    continue
                
                # Compute angle relative to keypoint
                theta = np.arctan2(da, dr)  # Angle in [-pi, pi]
                theta_normalized = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
                
                # Compute angular bin
                angular_bin = int(theta_normalized * alpha) % alpha
                
                # Compute radial bin (distance from keypoint)
                radial_bin = int((dist / max_radius) * rho)
                radial_bin = min(radial_bin, rho - 1)
                
                # Range weighting: weight by range to correct for range-density bias
                # Normalized range value (0 to 1)
                range_weight = r / num_ranges
                
                # Intensity weight
                intensity = img[a, r]
                
                # Combined weight
                weight = intensity * range_weight
                
                # Add to histograms
                angular_hist[angular_bin] += weight
                radial_hist[radial_bin] += weight
        
        # Process angular histogram: FFT and normalize phase
        # Take FFT
        fft_angular = np.fft.fft(angular_hist)
        # Get magnitude and phase
        magnitude = np.abs(fft_angular)
        phase = np.angle(fft_angular)
        
        # Normalize phase to [0, 1]
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        
        # Use magnitude for rotation invariance (alternative: use phase_normalized)
        # According to paper: "normalize its phase" - using phase
        angular_descriptor = phase_normalized[:alpha]
        
        # Normalize radial histogram
        radial_sum = np.sum(radial_hist)
        if radial_sum > 0:
            radial_descriptor = radial_hist / radial_sum
        else:
            radial_descriptor = radial_hist
        
        # Concatenate descriptors
        descriptor = np.concatenate([angular_descriptor, radial_descriptor])
        descriptors.append(descriptor)
    
    return np.array(descriptors)

#TODO: Check this implementation
def estimate_oriented_surface_points(keypoints, r=5.0, f=2.0, min_neighbors=6, 
                                      max_condition_number=1e5, z_min=None):
    """
    Estimate oriented surface points from keypoints with intensity information.
    
    Parameters:
    -----------
    keypoints : ndarray
        Keypoints with shape (N, 3) where each row is (x, y, intensity)
    r : float
        Radius in meters used for local neighborhood statistics
    f : float
        Re-sampling factor for downsampling grid side length (r/f)
    min_neighbors : int
        Minimum number of points in local neighborhood to keep a surface estimate
    max_condition_number : float
        Maximum allowed covariance condition number
    z_min : float or None
        Minimum intensity used in W_jj = z_j - z_min. If None, uses minimum from keypoints.
    
    Returns:
    --------
    oriented_points : ndarray
        Array of shape (M, 4) containing (mu_x, mu_y, n_x, n_y)
    """
    keypoints = np.asarray(keypoints, dtype=float)
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError(f"Expected keypoints with shape (N, 3), got {keypoints.shape}")
    if keypoints.size == 0:
        return np.empty((0, 4), dtype=float), np.empty((0, 2), dtype=float)

    # Extract positions and intensities
    positions = keypoints[:, :2]  # (N, 2)
    intensities = keypoints[:, 2]  # (N,)
    
    grid_size = r / f

    # 1) Downsample to grid centers
    cell_indices = np.floor(positions / grid_size).astype(int)
    cells = {}
    for idx, cell in enumerate(cell_indices):
        key = (int(cell[0]), int(cell[1]))
        if key not in cells:
            cells[key] = []
        cells[key].append(idx)
    
    pd = []
    for cell_x, cell_y in cells.keys():
        cell_center = np.array([(cell_x + 0.5) * grid_size, (cell_y + 0.5) * grid_size], dtype=float)
        pd.append(cell_center)

    if len(pd) == 0:
        return np.empty((0, 4), dtype=float), np.empty((0, 2), dtype=float)

    pd = np.asarray(pd, dtype=float)

    if z_min is None:
        z_min = float(np.min(intensities))

    # 2) For each grid center, estimate local distribution from neighbors
    kdtree = cKDTree(positions)
    oriented_points = []
    
    for pi in pd:
        neighbor_ids = np.asarray(kdtree.query_ball_point(pi, r), dtype=int)
        if neighbor_ids.size == 0:
            continue
        
        neighbors = positions[neighbor_ids]
        neighbor_intensities = intensities[neighbor_ids]

        if neighbors.shape[0] < min_neighbors:
            continue

        # Build weights from intensities
        weights = np.maximum(neighbor_intensities - z_min, 0.0)
        trace_w = float(np.sum(weights))
        if trace_w <= 0.0:
            weights = np.ones_like(weights, dtype=float) / len(weights)
        else:
            weights = weights / trace_w

        # Weighted statistics
        mu_i = np.sum(neighbors * weights[:, np.newaxis], axis=0)
        centered = neighbors - mu_i
        sigma_i = centered.T @ (weights[:, np.newaxis] * centered)

        eigenvalues, eigenvectors = np.linalg.eigh(sigma_i)
        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])

        if lambda_min <= 0:
            continue

        kappa = lambda_max / lambda_min
        if kappa > max_condition_number:
            continue

        n_i = eigenvectors[:, 0]
        n_i = n_i / (np.linalg.norm(n_i) + 1e-12)
        oriented_points.append([mu_i[0], mu_i[1], n_i[0], n_i[1]])

    if len(oriented_points) == 0:
        return np.empty((0, 4), dtype=float), pd

    return np.asarray(oriented_points, dtype=float), pd

def orb_descriptor(img, keypoints):
    """
    Compute ORB descriptors for given keypoints in the radar image.
    """
    # Convert keypoints to OpenCV format
    cv_keypoints = [cv2.KeyPoint(float(kp[1]), float(kp[0]), 1) for kp in keypoints]
    
    # Create ORB descriptor extractor
    orb = cv2.ORB_create()
    
    # Compute descriptors
    _, descriptors = orb.compute(img.astype(np.uint8), cv_keypoints)
    
    return descriptors

def radial_statistics_descriptor(keypoints, M:int=8, eps=1e-12):
    """
    Compute the Radial Statistics Descriptor (RSD) for the landmarks.
    """
    descriptors = np.empty((len(keypoints), 3, M), dtype=float)

    # Convert to Cartesian
    keypoints_cart = polar_to_cartesian_points(keypoints[:, 1], keypoints[:, 0])

    for idx in range(len(keypoints)):
        # Initialize RSD with zeros
        RSD = np.zeros((3, M), dtype=float)

        landmark = keypoints[idx]

        # Compute the distances from the landmark to all other keypoints in Cartesian space.
        deltas = keypoints_cart - landmark
        distances = np.linalg.norm(deltas, axis=1)

        # Exclude zero-distance points (typically the center point itself).
        valid = distances > eps
        if not np.any(valid):
            return RSD
        
        distances = distances[valid]

        # Angles in [0, 2pi), increasing CCW.
        angles = np.mod(np.arctan2(deltas[:, 1], deltas[:, 0]), 2.0 * np.pi)
        slice_width = 2.0 * np.pi / M
        slice_idx = np.floor(angles / slice_width).astype(int)
        slice_idx = np.clip(slice_idx, 0, M - 1)

        for j in range(M):
            Sj = distances[slice_idx == j]
            if Sj.size == 0:
                continue

            # Compute for each slice [|S|, arithmetic mean, harmonic mean].
            RSD[0, j] = float(Sj.size)
            RSD[1, j] = float(np.mean(Sj))
            RSD[2, j] = float(Sj.size / np.sum(1.0 / (Sj + eps)))

        # Normalize each row independently (L1, if non-zero).
        row_sums = np.sum(np.abs(RSD), axis=1, keepdims=True)
        non_zero_rows = row_sums[:, 0] > 0
        RSD[non_zero_rows] = RSD[non_zero_rows] / row_sums[non_zero_rows]

        # Rotate columns so highest-density slice is first.
        max_density_idx = int(np.argmax(RSD[0]))
        RSD = np.roll(RSD, -max_density_idx, axis=1)

        descriptors[idx] = RSD

    return np.array(descriptors)