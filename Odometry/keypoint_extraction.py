import numpy as np
from scipy.ndimage import prewitt
import matplotlib.pyplot as plt

from data_loading import load_radar_images, polar_to_cartesian_image

def compute_H_S(img, return_mag = False):
    prewitt_range = prewitt(img, axis=1)
    prewitt_angle = prewitt(img, axis=0)
    prewitt_mag = np.sqrt(prewitt_range**2 + prewitt_angle**2)
    prewitt_mag_norm = prewitt_mag / np.max(prewitt_mag)  # Normalize to [0, 1]
    S = img - np.mean(img)
    H = (1 - prewitt_mag_norm) * S
    if return_mag:
        return S, H, prewitt_mag_norm
    return S, H

def extract_keypoints(H, S, l_max = 500, return_mask = False):
    mask = np.zeros_like(H, dtype=bool) 
    keypoints = []
    l = 0
    counter = 0

    H_flat = H.flatten()
    sorted_indices = np.argsort(H_flat)[::-1]  # Indices of sorted values
    #H_sorted = H_flat[sorted_indices]  # Sorted values

    while l < len(sorted_indices) and counter < l_max:
        # Get the index of the l-th highest intensity point
        idx = sorted_indices[l]
        
        # Convert flat index back to 2D coordinates (a=angle/row, r=range/col)
        a, r = np.unravel_index(idx, H.shape)
        
        # If this point is not already masked, process it
        if not mask[a, r]:
            #keypoints.append((a, r))
            #print(f"Keypoint {len(keypoints)}: Angle index = {a}, Range index = {r}, Intensity = {H[a, r]:.2f}")
            
            # Search along angle 'a' for closest range indices with S < 0
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
            
            if np.all(mask[a, rlow:rhigh] == 0):
                counter += 1
                
            
            # Mark the region between rlow and rhigh at angle a
            mask[a, rlow:rhigh+1] = True
        
        l += 1  # Always increment l to move to next point

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
                keypoints.append((int(a), best_r))

    print(f"Found {len(keypoints)} keypoints")
    keypoints = np.array(keypoints)

    if return_mask:
        return keypoints, mask
    
    return keypoints

def k_strongest_keypoints(img, z_min, k=12):
    """
    Extract keypoints, by selecting the k strongest readings per azimuth that exceed a specified intensity threshold z_min.
    """
    keypoints = []
    for a in range(img.shape[0]):
        # Get the indices of the k strongest readings for this angle
        strongest_indices = np.argsort(img[a, :])[-k:]
        
        # Filter out readings that do not exceed the intensity threshold
        valid_indices = [r for r in strongest_indices if img[a, r] > z_min]
        
        # Add valid keypoints to the list
        for r in valid_indices:
            keypoints.append((a, r))
    
    return np.array(keypoints)

def motion_compensation(keypoints, ego_motion):
    """Apply motion compensation to keypoints based on ego-motion data.
    Assume that there is no acceleration.
    ego_motion = (v_x, v_y, theta) - velocity in x and y direction and angular velocity"""
    angle_resolution = 400 # 400 azimuth bins
    delta_T = 0.5 #s - time for a full radar scan

    time_offset = (keypoints[:, 0] - angle_resolution/2)*delta_T/2

    R = np.array([[np.cos(-ego_motion[2]), -np.sin(-ego_motion[2])],
                  [np.sin(-ego_motion[2]), np.cos(-ego_motion[2])]])
    trans = np.array([ego_motion[0], ego_motion[1]])
    
    compensated_keypoints = R @ keypoints - time_offset*trans

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

def compare_keypoint_methods(img):
    S, H = compute_H_S(img)
    keypoints_HS = extract_keypoints(H, S, l_max=500)
    keypoints_kstrongest = k_strongest_keypoints(img, z_min=0.6, k=12)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, aspect='auto')
    plt.scatter(keypoints_HS[:, 1], keypoints_HS[:, 0], c='red', s=20, marker='x')
    plt.title("Keypoints from H-S Method")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.subplot(1, 2, 2)
    plt.imshow(img, aspect='auto')
    plt.scatter(keypoints_kstrongest[:, 1], keypoints_kstrongest[:, 0], c='blue', s=20, marker='o')
    plt.title("Keypoints from K-Strongest Method")
    plt.xlabel("Range")
    plt.ylabel("Angle")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load radar image 
    image_file, image = load_radar_images(num_images=1)
    img = image[0]
    
    cartesian_image = polar_to_cartesian_image(img)
    
    S, H = compute_H_S(img)
    
    keypoints, mask = extract_keypoints(H, S, l_max=500, return_mask=True)
    
    #visualize_keypoints(img, cartesian_image, keypoints)
    visualize_kp_pipeline(img, keypoints, mask)
    compare_keypoint_methods(img)
