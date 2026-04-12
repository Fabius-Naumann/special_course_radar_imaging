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

def compute_H_S(img, return_mag = False):
    # Compute the gradient using the Prewitt operator in both directions
    prewitt_range = prewitt(img, axis=1)
    prewitt_angle = prewitt(img, axis=0)

    # Compute the magnitude of the gradient and normalize it
    prewitt_mag = np.sqrt(prewitt_range**2 + prewitt_angle**2)
    prewitt_mag_norm = prewitt_mag / np.max(prewitt_mag)  # Normalize to [0, 1]

    # S is the zero-mean image, H is the saliency map that combines intensity and gradient information
    S = img - np.mean(img)
    H = (1 - prewitt_mag_norm) * S

    # Optionally returns the normalized gradient magnitude for visualization
    if return_mag:
        return S, H, prewitt_mag_norm
    return S, H

def Cen2018_keypoints(img, w_median = 5, w_binom = 3, z_min = 0.6):
    """
    Extract keypoints using the Cen2018 landmark detection method.
    
    This method removes noise by:
    1. Subtracting noise floor (median filter) to get unbiased signal q
    2. Smoothing with binomial filter to get low-frequency signal p
    3. Estimating noise characteristics from negative q values
    4. Scaling by Gaussian probability to suppress noise
    5. Extracting peak centers as landmarks
    
    Parameters:
    -----------
    img : ndarray
        Radar image in polar coordinates (azimuth x range)
    w_median : int
        Window size for median filter
    w_binom : int
        Window size for binomial filter
    z_min : float
        Confidence threshold (z-score) for landmark detection
        
    Returns:
    --------
    keypoints : ndarray
        Array of (azimuth, range) tuples for detected landmarks
    """
    
    N = img.shape[1]  # Number of range bins
    keypoints = []
    
    # Create binomial filter kernel once (moved outside loop for performance)
    if w_binom == 3:
        kernel = np.array([1, 2, 1]) / 4.0
    elif w_binom == 5:
        kernel = np.array([1, 4, 6, 4, 1]) / 16.0
    elif w_binom == 7:
        kernel = np.array([1, 6, 15, 20, 15, 6, 1]) / 64.0
    else:
        # General binomial coefficients using Pascal's triangle
        from scipy.special import comb
        kernel = np.array([comb(w_binom-1, k) for k in range(w_binom)])
        kernel = kernel / kernel.sum()

    # Treat each power spectrum of each azimuth separately
    for a in range(img.shape[0]):
        # Step 1: Median Filtering to estimate and remove noise floor v(s)
        # This preserves high-frequency information
        img_med = median_filter(img[a], size=w_median)
        q = img[a] - img_med  # Unbiased signal with high-frequency info

        # Step 2: Binomial Filtering to smooth and obtain low-frequency signal p
        p = convolve(q, kernel, mode='same')

        # Step 3: Estimate noise characteristics
        # Treat values of q that fall below 0 as Gaussian noise with mean μq = 0
        Q_negative = q[q <= 0]
        
        if len(Q_negative) > 1:
            sigma_q = np.std(Q_negative, ddof=1)
        else:
            sigma_q = 1e-6  # Small value to avoid division by zero
        
        if sigma_q < 1e-10:  # Handle case of very small/zero noise
            sigma_q = 1e-6

        # Step 4: Compute weighted signal emphasizing peaks
        # Use inverse Gaussian weighting to emphasize outliers
        # For p: weight by how unusual this value is (z-score)
        z_score_p = np.abs(p) / sigma_q
        weight_p = np.maximum(0, z_score_p - z_min)
        
        # For q: weight by deviation from smoothed signal
        z_score_q = np.abs(q - p) / sigma_q  
        weight_q = np.maximum(0, z_score_q - z_min)
        
        # Combine: weight both components, emphasize high-frequency peaks
        y_hat = p * weight_p + (q - p) * weight_q
        
        # Step 5: Threshold to keep only significant peaks
        # Use more aggressive thresholding to avoid too many keypoints
        if np.any(y_hat > 0):
            # Use 95th percentile instead of 60th for stricter selection
            threshold = np.percentile(y_hat[y_hat > 0], 99.9)
            y_hat[y_hat < threshold] = 0
        else:
            y_hat[:] = 0
        
        # Step 6: Extract landmarks from peaks
        # Find all local maxima in y_hat more efficiently
        if np.any(y_hat > 0):
            # Find local maxima: points greater than both neighbors
            local_max = np.zeros(N, dtype=bool)
            local_max[1:-1] = (y_hat[1:-1] > y_hat[:-2]) & (y_hat[1:-1] > y_hat[2:]) & (y_hat[1:-1] > 0)
            # Handle edges
            if y_hat[0] > 0 and (N == 1 or y_hat[0] > y_hat[1]):
                local_max[0] = True
            if y_hat[-1] > 0 and (N == 1 or y_hat[-1] > y_hat[-2]):
                local_max[-1] = True
            
            # Get indices of local maxima
            peak_indices = np.where(local_max)[0]
            
            # Add all peaks as keypoints with intensity
            for peak_idx in peak_indices:
                intensity = y_hat[peak_idx]
                keypoints.append((a, int(peak_idx), intensity))
    
    print(f"Found {len(keypoints)} keypoints using Cen2018 method")
    return np.array(keypoints) if len(keypoints) > 0 else np.array([]).reshape(0, 3)

def Cen2019_keypoints(H, S, l_max = 500, return_mask = False):
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

def k_strongest_keypoints(img, z_min, k=12):
    """
    Extract keypoints, by selecting the k strongest readings per azimuth that exceed a specified intensity threshold z_min.
    Returns keypoints as (angle, range, intensity) tuples.
    """
    keypoints = []
    for a in range(img.shape[0]):
        # Get the indices of the k strongest readings for this angle
        strongest_indices = np.argsort(img[a, :])[-k:]
        
        # Filter out readings that do not exceed the intensity threshold
        valid_indices = [r for r in strongest_indices if img[a, r] > z_min]
        
        # Add valid keypoints to the list with intensity
        for r in valid_indices:
            intensity = img[a, r]
            keypoints.append((a, r, intensity))
    
    return np.array(keypoints)

def hessian_blob_keypoints(img, percentile=99.9, num_keypoints=300, sigma=2.0, is_cartesian=False):
    """
    Extract keypoints using Hessian-based blob detector with Adaptive Non-Maximal Suppression (ANMS).
    """
    # Hessian-based blob detection should run in Cartesian coordinates.
    if is_cartesian:
        img_cart = img
    else:
        img_cart = polar_to_cartesian_image(img)

    # Step 1: Smooth image with Gaussian filter
    img_smooth = gaussian_filter(img_cart.astype(float), sigma=sigma)
    
    # Step 2: Compute second-order derivatives for Hessian matrix
    # For 2D image: H = [[Ixx, Ixy], [Ixy, Iyy]]
    
    # First derivatives
    Ix = np.gradient(img_smooth, axis=1)
    Iy = np.gradient(img_smooth, axis=0)
    
    # Second derivatives
    Ixx = np.gradient(Ix, axis=1)
    Iyy = np.gradient(Iy, axis=0)
    Ixy = np.gradient(Ix, axis=0)
    
    # Step 3: Compute determinant of Hessian matrix as blob response
    # det(H) = Ixx * Iyy - Ixy^2
    hessian_response = Ixx * Iyy - Ixy**2
    
    # Step 4: Find candidate keypoints above threshold
    # Use absolute value to detect both bright and dark blobs
    hessian_response_abs = np.abs(hessian_response)
    
    # Adaptive thresholding: use percentile based on distribution
    max_response = np.max(hessian_response_abs)
    print(f"  Hessian response range: [{np.min(hessian_response_abs):.6f}, {max_response:.6f}]")
    
    # Use adaptive threshold based on distribution (99.5th percentile)
    if max_response > 0:
        adaptive_threshold = np.percentile(hessian_response_abs, percentile)
        print(f"  Using adaptive threshold: {adaptive_threshold:.6f}")
    else:
        print("  No valid Hessian responses found")
        return np.array([]).reshape(0, 2)
    
    # Get coordinates where response exceeds threshold
    candidate_coords = np.argwhere(hessian_response_abs > adaptive_threshold)
    
    if len(candidate_coords) == 0:
        print(f"  No keypoints found above threshold {adaptive_threshold:.6f}")
        return np.array([]).reshape(0, 3)
    
    print(f"  Found {len(candidate_coords)} candidates before ANMS")
    
    # Get response values for candidates
    candidate_responses = hessian_response_abs[candidate_coords[:, 0], candidate_coords[:, 1]]
    
    # Step 5: Apply Adaptive Non-Maximal Suppression (ANMS)
    keypoints_selected = anms(candidate_coords, candidate_responses, num_keypoints)
    
    # Add intensity (hessian response) as third column
    keypoints_intensities = hessian_response_abs[keypoints_selected[:, 0], keypoints_selected[:, 1]]
    keypoints = np.column_stack([keypoints_selected, keypoints_intensities])
    
    print(f"Found {len(keypoints)} keypoints using Hessian-ANMS method")
    return keypoints

def anms(coords, responses, num_retain):
    """
    Adaptive Non-Maximal Suppression (ANMS) for selecting spatially distributed keypoints.
    
    Based on: Bailo et al. (2018) "Efficient adaptive non-maximal suppression algorithms 
    for homogeneous spatial keypoint distribution"
    """
    n_candidates = len(coords)
    
    if n_candidates <= num_retain:
        return coords
    
    # For each point, find the minimum suppression radius
    # A point's suppression radius is the distance to the nearest stronger point
    suppression_radii = np.full(n_candidates, np.inf)
    
    # Sort by response strength (descending)
    sorted_indices = np.argsort(responses)[::-1]
    
    for i in range(n_candidates):
        idx_i = sorted_indices[i]
        coord_i = coords[idx_i]
        response_i = responses[idx_i]
        
        # Find minimum distance to any stronger point
        for j in range(i):  # Only check stronger points (earlier in sorted list)
            idx_j = sorted_indices[j]
            coord_j = coords[idx_j]
            
            # Compute Euclidean distance
            dist = np.sqrt(np.sum((coord_i - coord_j)**2))
            
            # Update suppression radius if this stronger point is closer
            if dist < suppression_radii[idx_i]:
                suppression_radii[idx_i] = dist
    
    # The strongest point has infinite radius (no stronger point exists)
    suppression_radii[sorted_indices[0]] = np.inf
    
    # Select top num_retain points with largest suppression radii
    # These are points that are locally strongest in their neighborhoods
    selected_indices = np.argsort(suppression_radii)[::-1][:num_retain]
    
    return coords[selected_indices]

def orb_keypoints(img, num_keypoints=300):
    """
    Extract keypoints using ORB detector on Cartesian-converted radar image.
    
    Parameters:
    -----------
    img : ndarray
        Radar image in polar coordinates (azimuth x range)
    num_keypoints : int
        Maximum number of keypoints to detect
        
    Returns:
    --------
    keypoints : ndarray
        Array of (azimuth, range) tuples in polar image indices
    """
    # ORB works best in Cartesian coordinates, so convert
    img_cart = polar_to_cartesian_image(img)
    cart_size = img_cart.shape[0]
    
    # Convert to uint8 (ORB expects 8-bit images)
    img_cart_uint8 = cv2.normalize(img_cart, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=num_keypoints)
    
    # Detect keypoints
    keypoints_cv = orb.detect(img_cart_uint8, None)
    
    # Convert keypoints from Cartesian pixel coordinates to polar indices
    # The Cartesian image maps: pixel (0,0) to (-1,-1) and (cart_size, cart_size) to (1,1)
    # in normalized coordinates
    keypoints = []
    num_angles, num_ranges = img.shape
    
    for kp in keypoints_cv:
        x_pix, y_pix = kp.pt  # Cartesian pixel coordinates
        
        # Convert pixel coordinates to normalized Cartesian coordinates [-1, 1]
        x_norm = (x_pix / cart_size) * 2.0 - 1.0
        y_norm = (y_pix / cart_size) * 2.0 - 1.0
        
        # Convert to polar coordinates
        r_norm = np.sqrt(x_norm**2 + y_norm**2)  # Normalized range [0, ~1.414]
        theta = np.arctan2(y_norm, x_norm)  # Angle in radians [-pi, pi]
        
        # Skip if outside unit circle (beyond valid radar range)
        if r_norm > 1.0:
            continue
        
        # Convert to polar image indices
        # Angle: map [-pi, pi] to [0, num_angles-1]
        theta_normalized = (theta + np.pi) / (2 * np.pi)  # [0, 1]
        a_idx = int(theta_normalized * num_angles) % num_angles
        
        # Range: map [0, 1] to [0, num_ranges-1]
        r_idx = int(r_norm * (num_ranges - 1))
        
        if 0 <= r_idx < num_ranges:
            # Get intensity from cartesian image at this location
            intensity = img_cart[int(y_pix), int(x_pix)]
            keypoints.append((a_idx, r_idx, intensity))
    
    print(f"Found {len(keypoints)} keypoints using ORB method")
    return np.array(keypoints) if len(keypoints) > 0 else np.array([]).reshape(0, 3)

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

def compare_keypoint_methods(img, cartesian_image, keypoints_HS, keypoints_2018, keypoints_kstrongest, keypoints_hessian, keypoints_orb):
    """Compare all keypoint detection methods with parameters tuned for similar output counts."""

    plt.figure(figsize=(24, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, aspect='auto')
    if len(keypoints_HS) > 0:
        plt.scatter(keypoints_HS[:, 1], keypoints_HS[:, 0], c='red', s=20, marker='x')
    plt.title(f"Cen2019 (H-S) Method ({len(keypoints_HS)} keypoints)")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.subplot(2, 3, 2)
    plt.imshow(img, aspect='auto')
    if len(keypoints_2018) > 0:
        plt.scatter(keypoints_2018[:, 1], keypoints_2018[:, 0], c='orange', s=20, marker='s')
    plt.title(f"Cen2018 Method ({len(keypoints_2018)} keypoints)")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.subplot(2, 3, 3)
    plt.imshow(img, aspect='auto')
    if len(keypoints_kstrongest) > 0:
        plt.scatter(keypoints_kstrongest[:, 1], keypoints_kstrongest[:, 0], c='blue', s=20, marker='o')
    plt.title(f"K-Strongest Method ({len(keypoints_kstrongest)} keypoints)")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.subplot(2, 3, 4)
    plt.imshow(
        cartesian_image,
        origin='lower',
        extent=[-1.0, 1.0, -1.0, 1.0],
        aspect='equal',
    )
    if len(keypoints_hessian) > 0:
        img_size = cartesian_image.shape[0]
        kp_x = (keypoints_hessian[:, 1] / img_size) * 2.0 - 1.0
        kp_y = (keypoints_hessian[:, 0] / img_size) * 2.0 - 1.0
        plt.scatter(kp_x, kp_y, c='green', s=20, marker='+')
    plt.title(f"Hessian-ANMS Method ({len(keypoints_hessian)} keypoints)")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.subplot(2, 3, 5)
    plt.imshow(img, aspect='auto')
    if len(keypoints_orb) > 0 and keypoints_orb.ndim == 2:
        plt.scatter(keypoints_orb[:, 1], keypoints_orb[:, 0], c='magenta', s=20, marker='d')
    plt.title(f"ORB Method ({len(keypoints_orb)} keypoints)")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load radar image 
    image_file, image = load_radar_images(num_images=1)
    img = image[0]
    
    print("Loaded radar image with shape:", img.shape)
    
    # Convert to cartesian for Hessian-based method
    cartesian_image = polar_to_cartesian_image(img)
    
    # Compute saliency maps for Cen2019 method
    S, H = compute_H_S(img)
    
    print("\n=== Testing Cen2019 Keypoint Extraction (H-S Method) ===")
    keypoints_2019, mask = Cen2019_keypoints(H, S, l_max=500, return_mask=True)
    
    print("\n=== Testing Cen2018 Keypoint Extraction ===")
    keypoints_2018 = Cen2018_keypoints(img, w_median=5, w_binom=3, z_min=0.5)
    
    print("\n=== Testing K-Strongest Keypoint Extraction ===")
    keypoints_kstrongest = k_strongest_keypoints(img, z_min=0.6, k=2)
    print(f"Found {len(keypoints_kstrongest)} keypoints using K-Strongest method")
    
    print("\n=== Testing Hessian-ANMS Keypoint Extraction ===")
    keypoints_hessian = hessian_blob_keypoints(
        cartesian_image,
        percentile=99.9,
        num_keypoints=300,
        is_cartesian=True,
    )
    
    print("\n=== Testing ORB Keypoint Extraction ===")
    keypoints_orb = orb_keypoints(img, num_keypoints=300)
    
    # Visualize Cen2019 pipeline
    #print("\n=== Visualizing Cen2019 Pipeline ===")
    #visualize_kp_pipeline(img, keypoints_2019, mask)
    
    # Compare all keypoint methods side by side
    print("\n=== Comparing All Keypoint Methods ===")
    compare_keypoint_methods(
        img,
        cartesian_image,
        keypoints_2019,
        keypoints_2018,
        keypoints_kstrongest,
        keypoints_hessian,
        keypoints_orb,
    )
    
    # Visualize individual methods on cartesian images
    print("\n=== Visualizing Keypoints on Cartesian Images ===")
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Convert keypoints from polar to cartesian for plotting
    num_angles, num_ranges = img.shape
    max_range = 1.0
    
    # Cen2019
    axes[0, 0].imshow(cartesian_image, origin='lower', extent=[-max_range, max_range, -max_range, max_range], aspect='equal')
    if len(keypoints_2019) > 0:
        kp = polar_to_cartesian_points(
            keypoints_2019[:, 1],
            keypoints_2019[:, 0],
            range_resolution=max_range / (num_ranges - 1),
            angle_resolution=2 * np.pi / (num_angles - 1),
        )
        axes[0, 0].scatter(kp[:,0], kp[:,1], c='red', s=20, marker='x', linewidths=2)
    axes[0, 0].set_title(f"Cen2019 (H-S) Method ({len(keypoints_2019)} keypoints)")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    
    # Cen2018
    axes[0, 1].imshow(cartesian_image, origin='lower', extent=[-max_range, max_range, -max_range, max_range], aspect='equal')
    if len(keypoints_2018) > 0:
        kp = polar_to_cartesian_points(
            keypoints_2018[:, 1],
            keypoints_2018[:, 0],
            range_resolution=max_range / (num_ranges - 1),
            angle_resolution=2 * np.pi / (num_angles - 1),
        )
        axes[0, 1].scatter(kp[:, 0], kp[:, 1], c='cyan', s=20, marker='o')
    axes[0, 1].set_title(f"Cen2018 Method ({len(keypoints_2018)} keypoints)")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    
    # K-Strongest
    axes[1, 0].imshow(cartesian_image, origin='lower', extent=[-max_range, max_range, -max_range, max_range], aspect='equal')
    if len(keypoints_kstrongest) > 0:
        kp = polar_to_cartesian_points(
            keypoints_kstrongest[:, 1],
            keypoints_kstrongest[:, 0],
            range_resolution=max_range / (num_ranges - 1),
            angle_resolution=2 * np.pi / (num_angles - 1),
        )
        axes[1, 0].scatter(kp[:,0], kp[:,1], c='yellow', s=20, marker='s')
    axes[1, 0].set_title(f"K-Strongest Method ({len(keypoints_kstrongest)} keypoints)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    
    # Hessian-ANMS (already in cartesian coordinates)
    axes[1, 1].imshow(cartesian_image, origin='lower', extent=[-max_range, max_range, -max_range, max_range], aspect='equal')
    if len(keypoints_hessian) > 0:
        # Convert from pixel coordinates to physical coordinates
        img_size = cartesian_image.shape[0]
        kp_x = (keypoints_hessian[:, 1] / img_size) * 2 * max_range - max_range
        kp_y = (keypoints_hessian[:, 0] / img_size) * 2 * max_range - max_range
        axes[1, 1].scatter(kp_x, kp_y, c='purple', s=20, marker='+', linewidths=2)
    axes[1, 1].set_title(f"Hessian-ANMS Method ({len(keypoints_hessian)} keypoints)")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")
    
    # ORB
    axes[1, 2].imshow(cartesian_image, origin='lower', extent=[-max_range, max_range, -max_range, max_range], aspect='equal')
    if len(keypoints_orb) > 0 and keypoints_orb.ndim == 2:
        kp = polar_to_cartesian_points(
            keypoints_orb[:, 1],
            keypoints_orb[:, 0],
            range_resolution=max_range / (num_ranges - 1),
            angle_resolution=2 * np.pi / (num_angles - 1),
        )
        axes[1, 2].scatter(kp[:, 0], kp[:, 1], c='magenta', s=20, marker='d')
    axes[1, 2].set_title(f"ORB Method ({len(keypoints_orb)} keypoints)")
    axes[1, 2].set_xlabel("X")
    axes[1, 2].set_ylabel("Y")
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Summary ===")
    print(f"Cen2019 (H-S):    {len(keypoints_2019)} keypoints")
    print(f"Cen2018:          {len(keypoints_2018)} keypoints")
    print(f"K-Strongest:      {len(keypoints_kstrongest)} keypoints")
    print(f"Hessian-ANMS:     {len(keypoints_hessian)} keypoints")
    print(f"ORB:              {len(keypoints_orb)} keypoints")
