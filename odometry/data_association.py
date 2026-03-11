import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

from utils.data_loading import load_radar_images, polar_to_cartesian_image, polar_to_cartesian_points
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

def estimate_oriented_surface_points(img, keypoints, r=5.0, f=2.0, min_neighbors=6, max_condition_number=1e5, z_min=None):
    """
    Estimate oriented surface points from a filtered radar point set.
    
    Parameters:
    -----------
    img : ndarray
        Radar image (angles x ranges), used to derive reflected intensities.
    keypoints : ndarray
        Array with shape (N, 2) where each row is (angle_idx, range_idx)
    r : float
        Radius in meters used for local neighborhood statistics
    f : float
        Re-sampling factor for downsampling grid side length (r/f)
    min_neighbors : int
        Minimum number of points in local neighborhood to keep a surface estimate
    max_condition_number : float
        Maximum allowed covariance condition number
    z_min : float or None
        Minimum intensity used in W_jj = z_j - z_min. If None, uses image minimum.
    
    Returns:
    --------
    oriented_points : ndarray
        Array of shape (M, 4) containing (mu_x, mu_y, n_x, n_y)
    """
    if keypoints is None or len(keypoints) == 0:
        return np.empty((0, 4), dtype=float)
    if img is None:
        raise ValueError("img must be provided to compute intensity-weighted covariance")

    # Convert (angle_idx, range_idx) points to Cartesian meters.
    pf = polar_to_cartesian_points(keypoints[:, 1], keypoints[:, 0])

    grid_size = r / f
    if grid_size <= 0:
        raise ValueError("r/f must be > 0")

    # 1) Downsample Pf to Pd by keeping one centroid per grid cell.
    cell_indices = np.floor(pf / grid_size).astype(int)
    cells = {}
    for idx, cell in enumerate(cell_indices):
        key = (int(cell[0]), int(cell[1]))
        if key not in cells:
            cells[key] = []
        cells[key].append(idx)

    pd = []
    for point_ids in cells.values():
        centroid = np.mean(pf[point_ids], axis=0)
        pd.append(centroid)

    if len(pd) == 0:
        return np.empty((0, 4), dtype=float)

    pd = np.asarray(pd, dtype=float)

    if z_min is None:
        z_min = float(np.min(img))

    # 2) For each pi in Pd, estimate local distribution from neighbors in Pf within radius r.
    oriented_points = []
    for pi in pd:
        diffs = pf - pi
        dists = np.linalg.norm(diffs, axis=1)
        neighbor_mask = dists <= r
        neighbors = pf[neighbor_mask]
        neighbor_ids = np.where(neighbor_mask)[0]

        # Discard ill-defined local distributions.
        if neighbors.shape[0] < min_neighbors:
            continue

        # Additional filtering step: discard neighborhoods from only one azimuth bin.
        neighbor_azimuths = keypoints[neighbor_ids, 0].astype(int)
        if np.unique(neighbor_azimuths).size <= 1:
            continue

        # Build weights W_jj = z_j - z_min and normalize by trace(W).
        neighbor_ranges = keypoints[neighbor_ids, 1].astype(int)
        neighbor_intensities = img[neighbor_azimuths, neighbor_ranges].astype(float)
        weights = np.maximum(neighbor_intensities - z_min, 0.0)
        trace_w = float(np.sum(weights))
        if trace_w <= 0.0:
            weights = np.ones_like(weights, dtype=float) / len(weights)
        else:
            weights = weights / trace_w

        # Weighted sample mean mu_i.
        mu_i = np.sum(neighbors * weights[:, np.newaxis], axis=0)
        centered = neighbors - mu_i

        # Weighted sample covariance: Sigma_i = (P-mu) W (P-mu)^T.
        # centered has shape (l, 2), so we use centered.T @ W @ centered -> (2, 2).
        sigma_i = centered.T @ (weights[:, np.newaxis] * centered)

        eigenvalues, eigenvectors = np.linalg.eigh(sigma_i)
        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])

        if lambda_min <= 0:
            continue

        kappa = lambda_max / lambda_min
        if kappa > max_condition_number:
            continue

        # 3) Normal is eigenvector corresponding to smallest eigenvalue.
        n_i = eigenvectors[:, 0]
        n_i = n_i / (np.linalg.norm(n_i) + 1e-12)
        oriented_points.append([mu_i[0], mu_i[1], n_i[0], n_i[1]])

    if len(oriented_points) == 0:
        return np.empty((0, 4), dtype=float)

    return np.asarray(oriented_points, dtype=float)

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

def compute_pairwiseCompatibilityScore(matches, keypoints1, keypoints2):
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

# Cost functions
def point_to_point_cost(mu1, mu2, R, t):
    # Apply rotation and translation to mu1
    mu1_transformed = R @ mu1 + t
    return np.linalg.norm(mu2 - mu1_transformed)

def point_to_line_cost(mu1, mu2, n2, R, t):
    """Point-to-line residual using target normal n2. Returns ||n^T · e||^2."""
    mu1_transformed = R @ mu1 + t
    e = mu1_transformed - mu2
    residual = float(n2.T @ e)
    return residual  # Return signed residual for optimization

def point_to_distribution_cost(mu1, mu2, sigma2, R, t, lambda_damp=0.1):
    """Point-to-distribution cost: e^T Σ^-1 e where Σ = (Σ + λI)."""
    mu1_transformed = R @ mu1 + t
    e = mu2 - mu1_transformed
    sigma_damped = sigma2 + lambda_damp * np.eye(sigma2.shape[0])
    return float(e.T @ np.linalg.inv(sigma_damped) @ e)

# Loss function
def Huber_loss(cost, delta):
    cost = abs(cost)
    if cost <= delta:
        return 0.5 * cost**2
    else:
        return delta * cost - 0.5 * delta**2

# Weight functions
def planarity(p):
    """Compute planarity: p = log(1 + |λmax/λmin|) as per CFEAR paper."""
    # p is a tuple with eigenvalues (lambda_min, lambda_max)
    lambda_min = abs(p[0]) + 1e-12
    lambda_max = abs(p[1]) + 1e-12
    return np.log(1.0 + abs(lambda_max / lambda_min))
    
def similarity(a, b):
    return 2*min(a, b) / (a + b + 1e-6)  # Add small epsilon to avoid division by zero

def planarity_similarity_weight(p1, p2):
    planarity1 = planarity(p1)
    planarity2 = planarity(p2)
    return similarity(planarity1, planarity2)

def detection_similarity_weight(d1, d2):
    return similarity(d1, d2)

def direction_weight(n1, n2):
    # n1 and n2 are normal vectors
    return max(0, np.dot(n1, n2))

def combined_weight(p1, p2, d1, d2, n1, n2):
    """Combined weight: w_i,j = w_plan + w_det + w_dir."""
    w_plan = planarity_similarity_weight(p1, p2)
    w_det = detection_similarity_weight(d1, d2)
    w_dir = direction_weight(n1, n2)
    return w_plan + w_det + w_dir


def _covariance_from_normal(n, sigma_along=2.0, sigma_across=0.15):
    """Build an anisotropic 2D covariance aligned with a surface normal."""
    n = np.asarray(n, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    t = np.array([-n[1], n[0]], dtype=float)  # tangent direction
    return (sigma_along**2) * np.outer(t, t) + (sigma_across**2) * np.outer(n, n)

def _rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _register_pair_cfear(
    oriented_points1,
    oriented_points2,
    max_iterations=20,
    distance_gate=6.0,
    theta_max_deg=45.0,
    huber_delta=0.5,
    cost_function="p2l",  # "p2p", "p2l", or "p2d"
    return_covariance=False,
):
    """
    Register two oriented point sets using CFEAR-3 approach.
    
    Corresponds to equation (5-6) in paper:
    arg min_xt fs2k(Mk, Mt, xt) = sum_{i,j in C} wi,j * Lδ(g(mk_j, mt_i, xt))
    
    Returns:
        R : rotation matrix (2x2)
        t : translation vector (2,)
        correspondences : list of (i, j) pairs
        covariance : 3x3 pose covariance (if return_covariance=True)
    """
    if oriented_points1 is None or oriented_points2 is None:
        return np.eye(2), np.zeros(2), [], None if return_covariance else (np.eye(2), np.zeros(2), [])
    if len(oriented_points1) == 0 or len(oriented_points2) == 0:
        return np.eye(2), np.zeros(2), [], None if return_covariance else (np.eye(2), np.zeros(2), [])

    oriented_points1 = np.asarray(oriented_points1, dtype=float)
    oriented_points2 = np.asarray(oriented_points2, dtype=float)

    theta = 0.0
    t = np.zeros(2, dtype=float)
    correspondences = []
    theta_max_rad = np.deg2rad(theta_max_deg)

    for iteration in range(max_iterations):
        R = _rotation_matrix(theta)

        # Find correspondences: nearest point within radius r with similar normal
        # Criterion: arccos(nj · ni) < θmax
        correspondences = []
        for i in range(oriented_points1.shape[0]):
            mu1 = oriented_points1[i, :2]
            n1 = oriented_points1[i, 2:]
            mu1_t = R @ mu1 + t
            n1_t = R @ n1

            best_j = -1
            best_dist = distance_gate + 1.0
            
            for j in range(oriented_points2.shape[0]):
                mu2 = oriented_points2[j, :2]
                n2 = oriented_points2[j, 2:]

                # Check distance gate
                dist = np.linalg.norm(mu2 - mu1_t)
                if dist > distance_gate:
                    continue

                # Check normal similarity: arccos(nj · ni) < θmax
                normal_dot = float(np.clip(np.dot(n1_t, n2), -1.0, 1.0))
                angle_diff = np.arccos(normal_dot)
                if angle_diff > theta_max_rad:
                    continue

                # Keep the closest valid match
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j != -1:
                correspondences.append((i, best_j))

        if len(correspondences) < 3:
            break

        # Minimize weighted scan-to-keyframe cost function (equation 6)
        H = np.zeros((3, 3), dtype=float)
        g = np.zeros(3, dtype=float)

        for i, j in correspondences:
            mu1 = oriented_points1[i, :2]
            n1 = oriented_points1[i, 2:]
            mu2 = oriented_points2[j, :2]
            n2 = oriented_points2[j, 2:]

            # Compute residual based on cost function (equations 12-14)
            sigma2 = _covariance_from_normal(n2)
            match cost_function:
                case "p2p":
                    residual = point_to_point_cost(mu1, mu2, R, t)
                    # Jacobian for P2P: de/dxt where e = ||error||
                    mu1_t = R @ mu1 + t
                    error_vec = mu1_t - mu2
                    error_norm = np.linalg.norm(error_vec) + 1e-12
                    dR_dtheta_mu = R @ np.array([-mu1[1], mu1[0]], dtype=float)
                    J = np.array([
                        2 * error_vec[0] / error_norm,
                        2 * error_vec[1] / error_norm,
                        2 * float(error_vec.T @ dR_dtheta_mu) / error_norm
                    ], dtype=float)
                case "p2l":
                    residual = point_to_line_cost(mu1, mu2, n2, R, t)
                    # Jacobian for P2L: d(n^T e)/dxt
                    dR_dtheta_mu = R @ np.array([-mu1[1], mu1[0]], dtype=float)
                    J = np.array([n2[0], n2[1], float(n2.T @ dR_dtheta_mu)], dtype=float)
                case "p2d":
                    residual = point_to_distribution_cost(mu1, mu2, sigma2, R, t)
                    # Jacobian for P2D: more complex, approximate with P2L form
                    dR_dtheta_mu = R @ np.array([-mu1[1], mu1[0]], dtype=float)
                    J = np.array([n2[0], n2[1], float(n2.T @ dR_dtheta_mu)], dtype=float)
                case _:
                    raise ValueError(f"Unknown cost_function: {cost_function}")

            # Apply Huber loss as weight (equation 7)
            huber_weight = Huber_loss(residual, huber_delta)

            # Compute similarity weights (equation 8-11)
            eigvals = np.linalg.eigvalsh(sigma2)
            p_feat = (float(eigvals[0]), float(eigvals[1]))
            # Use number of correspondences as proxy for detection count
            d1 = float(len(correspondences))
            d2 = float(len(correspondences))
            w_similarity = combined_weight(p_feat, p_feat, d1, d2, R @ n1, n2)

            # Combined weight: Huber * similarity
            w = huber_weight * w_similarity

            H += w * np.outer(J, J)
            g += w * J * residual

        # Solve for incremental update
        try:
            delta = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            break

        t += delta[:2]
        theta += delta[2]

        if np.linalg.norm(delta) < 1e-4:
            break

    final_R = _rotation_matrix(theta)
    
    # Estimate covariance from Hessian: C(xt) = (J^T J)^-1
    if return_covariance:
        try:
            covariance = np.linalg.inv(H + 1e-9 * np.eye(3))
        except np.linalg.LinAlgError:
            covariance = np.eye(3) * 1e6  # Large uncertainty if singular
        return final_R, t, correspondences, covariance
    
    return final_R, t, correspondences


def registration_from_oriented_points(oriented_points, window_size=5, max_iterations=20, return_covariance=False):
    """
    Estimate rotation and translation using CFEAR-style oriented points.
    
    Parameters:
    -----------
    oriented_points : list of ndarray or tuple(ndarray, ndarray)
        If list: ordered scans, each with shape (N, 4) containing (mu_x, mu_y, n_x, n_y).
        If tuple/list of length 2: register first set to second set.
    return_covariance : bool
        If True, return pose covariance estimate from Hessian

    Returns:
    --------
    If two sets are provided:
        R : ndarray
            Estimated rotation matrix (2x2)
        t : ndarray
            Estimated translation vector (2,)
        [covariance] : ndarray (optional)
            3x3 pose covariance if return_covariance=True
    If a scan list is provided:
        transforms : list of tuples
            Pairwise transforms [(R_0_1, t_0_1, [cov_0_1]), ...]
    """
    # Pair input: directly estimate one transform.
    if isinstance(oriented_points, tuple) and len(oriented_points) == 2:
        result = _register_pair_cfear(
            oriented_points[0],
            oriented_points[1],
            max_iterations=max_iterations,
            return_covariance=return_covariance,
        )
        return result

    if isinstance(oriented_points, list) and len(oriented_points) == 2 and isinstance(oriented_points[0], np.ndarray):
        result = _register_pair_cfear(
            oriented_points[0],
            oriented_points[1],
            max_iterations=max_iterations,
            return_covariance=return_covariance,
        )
        return result

    # Scan list input: estimate pairwise transforms and optionally fuse over a local window.
    if not isinstance(oriented_points, list) or len(oriented_points) < 2:
        return []

    transforms = []
    for k in range(len(oriented_points) - 1):
        result = _register_pair_cfear(
            oriented_points[k],
            oriented_points[k + 1],
            max_iterations=max_iterations,
            return_covariance=return_covariance,
        )
        if return_covariance:
            R_k, t_k, _, cov_k = result
            transforms.append((R_k, t_k, cov_k))
        else:
            R_k, t_k, _ = result
            transforms.append((R_k, t_k))

    # Simple window smoothing on translation for stability.
    if window_size > 1 and len(transforms) > 0:
        smoothed = []
        for i in range(len(transforms)):
            start = max(0, i - window_size + 1)
            if return_covariance:
                t_stack = np.array([transforms[j][1] for j in range(start, i + 1)])
                t_mean = np.mean(t_stack, axis=0)
                smoothed.append((transforms[i][0], t_mean, transforms[i][2]))
            else:
                t_stack = np.array([transforms[j][1] for j in range(start, i + 1)])
                t_mean = np.mean(t_stack, axis=0)
                smoothed.append((transforms[i][0], t_mean))
        transforms = smoothed

    return transforms

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
