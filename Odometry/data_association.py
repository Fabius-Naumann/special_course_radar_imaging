import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

from data_loading import load_radar_images, polar_to_cartesian_image
from keypoint_extraction import compute_H_S, extract_keypoints, visualize_keypoints

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

def unaryMatchesFromDescriptors(desc1, desc2):
    """
    Compute unary matches between two sets of descriptors
    
    Parameters:
    -----------
    desc1 : ndarray
        Descriptors for keypoints in image 1 (shape: N1 x D)
    desc2 : ndarray
        Descriptors for keypoints in image 2 (shape: N2 x D)

    
    Returns:
    --------
    matches : list of tuples
        List of matched keypoint indices (idx1, idx2)
    """
    # Match for each keypoint of L1 a keypoint of L2 based on the descriptor
    matches = []

    if desc1.ndim == 1:
        desc1 = desc1.reshape(1, -1)
    if desc2.ndim == 1:
        desc2 = desc2.reshape(1, -1)

    for i, desc1_vec in enumerate(desc1):
        best_match_idx = -1
        best_distance = float('inf')

        for j, desc2_vec in enumerate(desc2):
            # Compute distance (e.g., Euclidean)
            distance = np.linalg.norm(desc1_vec - desc2_vec)

            if distance < best_distance:
                best_distance = distance
                best_match_idx = j

        #print(f"Keypoint {i} in Image 1 matches with Keypoint {best_match_idx} in Image 2 (Distance: {best_distance:.4f})")
        matches.append((i, best_match_idx, best_distance))

    return matches

def compute_pairwiseCompatibilityScore(matches,keypoints1, keypoints2):
    """
    Compute pairwise compatibility score for matches
    
    Parameters:
    -----------
    matches : list of tuples
        List of matched keypoint indices (idx1, idx2)
    keypoints1 : ndarray
        Keypoints in image 1 (shape: N1 x 2)
    keypoints2 : ndarray
        Keypoints in image 2 (shape: N2 x 2)

    Returns:
    --------
    compatibility_scores : 
        Values as compatibility scores
    """
    # Compute compatibility score for each pair of proposed matches g=(i,i') and h=(j,j')
    # if matches g and h are correct then the relationship between i and j is similar to i' and j'
    # computed by the distances between corresponding pairs of points in the two images

    compatibility_scores = np.zeros((len(matches), len(matches)))

    for g in range(len(matches)):
        i, i_prime, dist_g = matches[g]
        kp_i = keypoints1[i]
        kp_i_prime = keypoints2[i_prime]
        
        for h in range(g + 1, len(matches)):
            j, j_prime, dist_h = matches[h]
            kp_j = keypoints1[j]
            kp_j_prime = keypoints2[j_prime]
            
            # Compute distances in image 1 and image 2
            dist_ij = np.linalg.norm(kp_i - kp_j)
            dist_i_prime_j_prime = np.linalg.norm(kp_i_prime - kp_j_prime)
            
            # Compute compatibility score (e.g., inverse of distance difference)
            if dist_ij > 0:
                compatibility_score = 1 / (1 + abs(dist_ij - dist_i_prime_j_prime))
            else:
                compatibility_score = 0
            
            compatibility_scores[g, h] = compatibility_score
            compatibility_scores[h, g] = compatibility_score  # Symmetric

    return compatibility_scores 

def select_matches(matches, Compatibility):
    eigenvalues, eigenvectors = eigs(Compatibility, k=1, which='LM')
    v_star = np.real(eigenvectors[:, 0])  # Principal eigenvector
    v_star = np.abs(v_star)  # Use absolute values for scores

    U = np.array(matches)  # U is u×3 array with (i, i', distance)
    M = []  # Selected matches
    m_hat = np.zeros(len(U))  # Binary vector indicating selected matches
    unsearched = set(range(len(U)))  # Indices of matches not yet processed
    score = 0

    print(f"\nStarting greedy match selection from {len(U)} candidate matches...")

    iteration = 0
    while len(unsearched) > 0:
        iteration += 1
        
        # Step 7: Find g with highest eigenvector value among unsearched
        max_score = -np.inf
        m_g = -1
        for g in unsearched:
            if v_star[g] > max_score:
                max_score = v_star[g]
                m_g = g
        
        if m_g == -1:
            break
        
        # Step 8: Compute current score
        current_score = m_hat.T @ Compatibility @ m_hat
        
        # Terminate if adding this match doesn't improve the score
        # (or if score would be negative)
        if current_score < score and len(M) > 0:
            print(f"Iteration {iteration}: Terminating - score not improving (current: {current_score:.4f}, previous: {score:.4f})")
            break
        
        # Step 9: Add the match to selected set
        M.append(U[m_g])
        m_hat[m_g] = 1
        score = current_score
        
        # Step 10: Remove conflicting matches from unsearched
        # A match h conflicts with g if they share a keypoint
        to_remove = set()
        for h in unsearched:
            # Check if match g and match h share any keypoint
            # U[g, 0] is keypoint index in L1, U[g, 1] is keypoint index in L2
            if U[m_g, 0] == U[h, 0] or U[m_g, 1] == U[h, 1]:
                to_remove.add(h)
        
        unsearched -= to_remove
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Selected {len(M)} matches, {len(unsearched)} candidates remaining")

    print(f"\nFinal: Selected {len(M)} matches from {len(U)} candidates")
    print(f"Final global compatibility score: {score:.4f}")

    # Convert M to numpy array for easier handling
    M = np.array(M)
    print(f"Selected matches shape: {M.shape}")
    return M

def visualize_matches(img1, img2, keypoints1, keypoints2, matches):
    # Visualize the matched keypoints between the two images with different colors for each match
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Generate colors for each match
    colors = plt.cm.hsv(np.linspace(0, 1, len(matches)))

    # Show first image with matched keypoints
    axes[0].imshow(img1, aspect='auto')
    for idx, match in enumerate(matches):
        kp1_idx = int(match[0])
        kp1 = keypoints1[kp1_idx]
        axes[0].scatter(kp1[1], kp1[0], c=[colors[idx]], s=50, marker='o')
    axes[0].set_title(f"Image 1 - {len(matches)} Matched Keypoints")
    axes[0].set_xlabel("Range")
    axes[0].set_ylabel("Angle")

    # Show second image with matched keypoints
    axes[1].imshow(img2, aspect='auto')
    for idx, match in enumerate(matches):
        kp2_idx = int(match[1])
        kp2 = keypoints2[kp2_idx]
        axes[1].scatter(kp2[1], kp2[0], c=[colors[idx]], s=50, marker='o')
    axes[1].set_title(f"Image 2 - {len(matches)} Matched Keypoints")
    axes[1].set_xlabel("Range")
    axes[1].set_ylabel("Angle")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_file, image = load_radar_images(num_images=2)
    img1 = image[0]
    img2 = image[1]
    
    cartesian_image1 = polar_to_cartesian_image(img1)
    cartesian_image2 = polar_to_cartesian_image(img2)
    
    S1, H1 = compute_H_S(img1)
    S2, H2 = compute_H_S(img2)
    
    keypoints1 = extract_keypoints(H1, S1, l_max=100)
    keypoints2 = extract_keypoints(H2, S2, l_max=100)
    
    visualize_keypoints(img1, cartesian_image1, keypoints1)
    visualize_keypoints(img2, cartesian_image2, keypoints2)

    descriptors1 = compute_descriptors(img1, keypoints1)
    descriptors2 = compute_descriptors(img2, keypoints2)

    U = unaryMatchesFromDescriptors(descriptors1, descriptors2)
    print(f"Found {len(U)} unary matches between the two images")
    Compatibility = compute_pairwiseCompatibilityScore(U, keypoints1, keypoints2)
    M = select_matches(U, Compatibility)

    visualize_matches(img1, img2, keypoints1, keypoints2, M)
