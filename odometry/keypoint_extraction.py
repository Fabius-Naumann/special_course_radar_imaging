import numpy as np
from scipy.ndimage import prewitt, median_filter, gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.stats import norm
import cv2

import sys
from pathlib import Path

# Add the project root folder to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils.data_loading import load_radar_images, polar_to_cartesian_image, polar_to_cartesian_points, cartesian_to_polar_points
from utils.cfar import _normalize_azimuth_rows, cfar2d_polar_ca, suggest_default_params

def compute_H_S(img, return_mag = False):
    """
    Compute Saliency (H) and Zero-Mean (S) maps from a radar image as per Cen2019.
    
    Parameters:
    - img: Raw radar image (polar or cartesian)
    - return_mag: Whether to return the gradient magnitude map as well
    
    Returns:
    - S: Zero-mean image
    - H: Saliency map combining intensity and suppressed gradient
    - prewitt_mag_norm (optional): normalized gradient magnitude
    """
    if img.size == 0:
        empty = np.asarray(img, dtype=float)
        if return_mag:
            return empty, empty, empty
        return empty, empty

    # Compute the gradient using the Prewitt operator in both directions
    prewitt_range = prewitt(img, axis=1)
    prewitt_angle = prewitt(img, axis=0)

    # Compute the magnitude of the gradient and normalize it
    prewitt_mag = np.sqrt(prewitt_range**2 + prewitt_angle**2)
    prewitt_max = float(np.max(prewitt_mag))
    if prewitt_max > 0.0:
        prewitt_mag_norm = prewitt_mag / prewitt_max  # Normalize to [0, 1]
    else:
        prewitt_mag_norm = np.zeros_like(prewitt_mag)

    # S is the zero-mean image, H is the saliency map that combines intensity and gradient information
    S = img - np.mean(img)
    H = (1 - prewitt_mag_norm) * S

    # Optionally returns the normalized gradient magnitude for visualization
    if return_mag:
        return S, H, prewitt_mag_norm
    return S, H

def Cen2019_keypoints(H, S, l_max = 500, return_mask = False):
    """
    Extract keypoints using the Cen2019 landmark detection method.
    
    Parameters:
    -----------
    H : ndarray
        Saliency map.
    S : ndarray
        Zero-mean image.
    l_max : int
        Maximum number of keypoints.
    return_mask : bool
        Whether to return the mask.
        
    Returns:
    --------
    keypoints : ndarray
        Extracted keypoints.
    mask : ndarray (optional)
        Mask of selected regions.
    """
    mask = np.zeros_like(H, dtype=bool) 
    keypoints = []
    l = 0
    counter = 0

    # Get indices in descending order
    H_flat = H.flatten()
    sorted_indices = np.argsort(H_flat)[::-1]  # Indices of sorted values
    #H_sorted = H_flat[sorted_indices]  # Sorted values

    while l < len(sorted_indices) and counter < l_max:
        # Get the index of the l-th highest intensity point
        idx = sorted_indices[l]
        
        # Get the corresponding angle and range of the point
        a, r = np.unravel_index(idx, H.shape)
        
        # If this point is not already masked, process it
        if not mask[a, r]:
            # Find range boundaries where S < 0 along the same angle 'a'      

            # Search below r for rlow
            rlow = 0
            for i in range(r - 1, -1, -1):
                if S[a, i] < 0:
                    rlow = i
                    break
            
            # Search above r for rhigh
            rhigh = S.shape[1] - 1
            for i in range(r + 1, S.shape[1]):
                if S[a, i] < 0:
                    rhigh = i
                    break
            
            # If that area wasn't masked yet, increase the counter
            if np.all(mask[a, rlow:rhigh] == 0):
                counter += 1
                
            # Mark the region between rlow and rhigh at angle a in the mask
            mask[a, rlow:rhigh+1] = True
        
        l += 1  # Always increment l to move to next point

    # After processing, extract the keypoints from the mask
    for a in range(mask.shape[0]):
        # Find continuous marked regions in this angle
        marked_cols = np.where(mask[a, :])[0]
        
        if len(marked_cols) == 0:
            continue
        
        # Find continuous regions (groups of consecutive columns)
        regions = []
        current_region = [marked_cols[0]]
        
        for i in range(1, len(marked_cols)):
            if marked_cols[i] == marked_cols[i-1] + 1:
                # Continuous, add to current region
                current_region.append(marked_cols[i])
            else:
                # Discontinuity, save current region and start new one
                regions.append(current_region)
                current_region = [marked_cols[i]]
        regions.append(current_region)  # Add last region
        
        # For each region, check if it has neighbors in adjacent angles
        for region in regions:
            # Check if adjacent angles have any marked regions
            has_neighbor = False
            for da in [-1, 1]:  # Check angle above and below
                neighbor_a = (a + da) % mask.shape[0]
                # Check if any marked cell exists at neighbor angle within region range
                region_min, region_max = min(region), max(region)
                if np.any(mask[neighbor_a, region_min:region_max+1]):
                    has_neighbor = True
                    break
            
            # If region has neighbors in azimuth, select the highest intensity point
            if has_neighbor:
                max_idx = np.argmax([H[a, r] for r in region])
                best_r = region[max_idx]
                intensity = H[a, best_r]
                keypoints.append((int(a), best_r, intensity))

    print(f"Found {len(keypoints)} keypoints")
    keypoints = np.array(keypoints) if len(keypoints) > 0 else np.array([]).reshape(0, 3)

    if return_mask:
        return keypoints, mask
    
    return keypoints

def preprocessing_normalized_azimuths(img):
    normalized_image = _normalize_azimuth_rows(img)
    return normalized_image

def preprocessing_cfar(img):
    # Implementation for CFAR preprocessing
    CFAR_PFA = 2e-1
    RANGE_BIN_M = 0.155
    params = suggest_default_params(range_bin_m=RANGE_BIN_M)

    det_ca, thr_ca, noise_ca, valid_ca = cfar2d_polar_ca(
            img,
            **(params | {"normalize_azimuth": True}),
            range_pad_mode="edge",
            pfa=CFAR_PFA,
            min_noise_floor_factor=1.3,
        )

    img_cfar = img * det_ca
    return img_cfar

def k_strongest_keypoints(img, z_min, k=12, max_distance_percentile=100):
    """
    Extract keypoints, by selecting the k strongest readings per azimuth that exceed a specified intensity threshold z_min.
    Returns keypoints as (angle, range, intensity) tuples.
    """
    keypoints = []
    max_range = img.shape[1]
    max_distance = max_range * max_distance_percentile / 100

    for a in range(img.shape[0]):
        # Get the indices of the k strongest readings for this angle
        strongest_indices = np.argsort(img[a, :])[-k:]
        
        # Filter out readings that do not exceed the intensity threshold
        valid_indices = [r for r in strongest_indices if img[a, r] > z_min]

        # Filter out readings that are beyond the maximum distance
        valid_indices = [r for r in valid_indices if r <= max_distance]
        
        # Add valid keypoints to the list with intensity
        for r in valid_indices:
            intensity = img[a, r]
            keypoints.append((a, r, intensity))
    
    return np.array(keypoints)

def motion_compensation(keypoints_polar, ego_motion, azimuth_bins=400, delta_T=0.5):
    """Apply motion compensation to keypoints based on ego-motion data.
    Assume that there is no acceleration.
    ego_motion = (v_x, v_y, theta) - velocity in x and y direction and angular velocity

    Parameters
    ----------
    keypoints : ndarray of shape (N, 2)
        Keypoints in cartesian coordinates (meters for cartesian).
    ego_motion : tuple
        (v_x, v_y, omega) where v is in m/s and omega is in rad/s.
    azimuth_bins : int
        Number of azimuth bins (for polar coordinate system only).
    delta_T : float
        Total scan duration in seconds.
    coordinate_system : str
        "polar" (default) uses azimuth-dependent time offsets,
        "cartesian" applies a uniform half-scan offset.
    """

    v_x, v_y, omega = ego_motion

    R = np.array([[np.cos(-omega), -np.sin(-omega)], [np.sin(-omega), np.cos(-omega)]], dtype=float)
    time_offsets = (keypoints_polar[:,0] - azimuth_bins / 2)* delta_T / 2

    keypoints = polar_to_cartesian_points(ranges=keypoints_polar[:,1], angles=keypoints_polar[:,0])

    compensated_keypoints = []
    for i, (a, r) in enumerate(keypoints_polar):
        # Compute the time offset for this keypoint based on its azimuth
        time_offset = time_offsets[i]

        # Compute the translation due to linear motion
        translation = np.array([v_x * time_offset, v_y * time_offset])

        # Compute the rotation due to angular motion
        rotation = R @ keypoints_polar[i]

        # Apply compensation: first rotate, then translate
        compensated_point = rotation + translation
        compensated_keypoints.append(compensated_point)
    return compensated_keypoints

def visualize_kp_pipeline(img, keypoints, mask):
    H, S, gradient = compute_H_S(img, True)

    fig = plt.figure(figsize=(25, 5))
    plt.subplots_adjust(left=0.05, right=0.98, wspace=0.35, hspace=0.3)
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, aspect='auto')
    plt.title("Original Radar Image")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.subplot(1, 3, 2)
    plt.imshow(gradient, aspect='auto')
    plt.title("Normalized Gradient Magnitude")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.subplot(1, 3, 3)
    plt.imshow(H, aspect='auto')
    #plt.scatter(keypoints[:, 1], keypoints[:, 0], c='red', s=10, marker='o')
    plt.title("Saliency Map (H)")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.show()

    fig = plt.figure(figsize=(25, 5))
    plt.subplots_adjust(left=0.05, right=0.98, wspace=0.35, hspace=0.3)
    
    plt.subplot(1, 2, 1)
    plt.imshow(mask, aspect='auto')
    plt.title("Masked Regions for Keypoint Selection")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.subplot(1 , 2, 2)
    plt.imshow(img, aspect='auto')
    plt.scatter(keypoints[:, 1], keypoints[:, 0], c='red', s=10, marker='o')
    plt.title("Selected Keypoints")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.show()

def visualize_keypoints(img_polar, img_cartesian, keypoints, max_range=1.0):
    num_angles, num_ranges = img_polar.shape

    # Show keypoints on the original image and on cartesian image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_polar, aspect='auto')
    plt.scatter(keypoints[:, 1], keypoints[:, 0], c='red', s=50, marker='x')  # Note: (r, a) = (col, row)
    plt.title("Keypoints on Original Radar Image")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.subplot(1, 2, 2)
    plt.imshow(
        img_cartesian,
        origin='lower',
        extent=[-max_range, max_range, -max_range, max_range],
        aspect='equal',
    )
    # Convert keypoints from polar to cartesian for plotting
    keypoint_angles = (keypoints[:, 0] / (num_angles - 1)) * 2 * np.pi
    keypoint_ranges = (keypoints[:, 1] / (num_ranges - 1)) * max_range
    keypoints_x = keypoint_ranges * np.cos(keypoint_angles)
    keypoints_y = keypoint_ranges * np.sin(keypoint_angles)
    plt.scatter(keypoints_x, keypoints_y, c='red', s=50, marker='x')
    plt.title("Keypoints on Cartesian Transformed Image")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load radar image 
    image_file, image = load_radar_images(num_images=1)
    img = image[0]
    
    print("\nLoaded radar image with shape:", img.shape)
    
    # Compute saliency maps for Cen2019 method
    S, H = compute_H_S(img)
    
    print("\n=== Testing Cen2019 Keypoint Extraction (H-S Method) ===")
    keypoints_2019, mask = Cen2019_keypoints(H, S, l_max=500, return_mask=True)
    
    print("\n=== Testing K-Strongest Keypoint Extraction ===")
    keypoints_kstrongest = k_strongest_keypoints(img, z_min=0.6, k=2)
    print(f"Found {len(keypoints_kstrongest)} keypoints using K-Strongest method")
    
    # Visualize Cen2019 pipeline
    print("\n=== Visualizing Cen2019 Pipeline ===")
    visualize_kp_pipeline(img, keypoints_2019, mask)
    
    print("\n=== Summary ===")
    print(f"Cen2019 (H-S):    {len(keypoints_2019)} keypoints")
    print(f"K-Strongest:      {len(keypoints_kstrongest)} keypoints")
