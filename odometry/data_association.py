import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import open3d as o3d

import sys
from pathlib import Path

# Add the project root folder to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loading import load_radar_images, polar_to_cartesian_image, polar_to_cartesian_points
from keypoint_extraction import compute_H_S, Cen2019_keypoints, visualize_keypoints
from descriptors import compute_descriptors

def ICP_registration(points1, points2, max_iterations=20, distance_threshold=0.5, R_init=np.eye(2), t_init=np.zeros(2)):
    # Create Open3D point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2.points = o3d.utility.Vector3dVector(points2)

    # Convert 2D transformation to 4x4 homogeneous transformation matrix for Open3D
    R = R_init
    t = t_init
    transformation_init = np.eye(4, dtype=float)
    transformation_init[:2, :2] = R
    transformation_init[:2, 3] = t

    # Initial alignment using Open3D's ICP
    threshold = distance_threshold
    reg = o3d.pipelines.registration.registration_icp(pcd1, pcd2, threshold, transformation_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    return reg.transformation

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

#TODO: Check this implementation
def _register_pair_cfear(
    oriented_points1,
    oriented_points2,
    max_iterations=20,
    distance_gate=6.0,
    theta_max_deg=45.0,
    huber_delta=0.5,
    cost_function="p2l",  # "p2p", "p2l", or "p2d"
    return_covariance=False,
    start_theta=0.0,
    start_t=np.zeros(2, dtype=float)
):
    """
    Register two oriented point sets using CFEAR-3 approach.
    Optimized version with vectorized operations and precomputation.
    
    Corresponds to equation (5-6) in paper:
    arg min_xt fs2k(Mk, Mt, xt) = sum_{i,j in C} wi,j * Lδ(g(mk_j, mt_i, xt))
    
    Returns:
        R : rotation matrix (2x2)
        t : translation vector (2,)
        correspondences : list of (i, j) pairs
        covariance : 3x3 pose covariance (if return_covariance=True)
    """
    if oriented_points1 is None or oriented_points2 is None:
        if return_covariance:
            return np.eye(2), np.zeros(2), [], np.eye(3) * 1e6
        return np.eye(2), np.zeros(2), []
    if len(oriented_points1) == 0 or len(oriented_points2) == 0:
        if return_covariance:
            return np.eye(2), np.zeros(2), [], np.eye(3) * 1e6
        return np.eye(2), np.zeros(2), []

    oriented_points1 = np.asarray(oriented_points1, dtype=float)
    oriented_points2 = np.asarray(oriented_points2, dtype=float)

    theta = start_theta
    t = start_t
    correspondences = []
    theta_max_rad = np.deg2rad(theta_max_deg)
    last_hessian = np.eye(3, dtype=float) * 1e6
    
    # Precompute covariances and eigenvalues for set 2 (constant during iterations)
    n_points2 = oriented_points2.shape[0]
    sigma2_list = []
    eigvals2_list = []
    for j in range(n_points2):
        n2 = oriented_points2[j, 2:]
        sigma2 = _covariance_from_normal(n2)
        sigma2_list.append(sigma2)
        eigvals2 = np.linalg.eigvalsh(sigma2)
        eigvals2_list.append(eigvals2)
    
    n_points1 = oriented_points1.shape[0]
    cos_theta_max = np.cos(theta_max_rad)  # Use dot product threshold instead of arccos
    distance_gate_sq = distance_gate ** 2  # Use squared distances
    
    for iteration in range(max_iterations):
        c = np.cos(theta)  # Precompute cos/sin
        s = np.sin(theta)
        # R = [[c, -s], [s, c]]
        
        # Vectorized transformation for set 1
        mu1_all = oriented_points1[:, :2]  # (n_points1, 2)
        n1_all = oriented_points1[:, 2:]   # (n_points1, 2)
        
        # Transform mu1: R @ mu1 + t
        mu1_t_all = np.array([c * mu1_all[:, 0] - s * mu1_all[:, 1],
                              s * mu1_all[:, 0] + c * mu1_all[:, 1]], dtype=float).T + t  # (n_points1, 2)
        
        # Transform normals: R @ n1
        n1_t_all = np.array([c * n1_all[:, 0] - s * n1_all[:, 1],
                             s * n1_all[:, 0] + c * n1_all[:, 1]], dtype=float).T  # (n_points1, 2)
        
        mu2_all = oriented_points2[:, :2]  # (n_points2, 2)
        n2_all = oriented_points2[:, 2:]   # (n_points2, 2)

        # Find correspondences using vectorized operations
        correspondences = []
        
        # Vectorized distance computation: (n_points1, n_points2)
        # dist[i,j] = ||mu2[j] - mu1_t[i]||^2
        diff = mu2_all[np.newaxis, :, :] - mu1_t_all[:, np.newaxis, :]  # (n_points1, n_points2, 2)
        dist_sq = np.sum(diff**2, axis=2)  # (n_points1, n_points2)
        
        # Vectorized normal similarity: (n_points1, n_points2)
        # dot product n1_t[i] · n2[j]
        normal_dots = np.dot(n1_t_all, n2_all.T)  # (n_points1, n_points2)
        
        # Find best match for each point in set 1
        for i in range(n_points1):
            # Apply distance gate and normal similarity filters
            valid_mask = (dist_sq[i, :] <= distance_gate_sq) & (normal_dots[i, :] >= cos_theta_max)
            
            if np.any(valid_mask):
                # Among valid matches, select closest
                valid_indices = np.where(valid_mask)[0]
                best_j = valid_indices[np.argmin(dist_sq[i, valid_indices])]
                correspondences.append((i, best_j))

        if len(correspondences) < 3:
            break

        # Optimize: vectorized computation of residuals and Jacobians
        H = np.zeros((3, 3), dtype=float)
        g = np.zeros(3, dtype=float)
        
        num_corr = len(correspondences)
        d_count = float(num_corr)
        
        for i, j in correspondences:
            mu1 = oriented_points1[i, :2]
            n1 = oriented_points1[i, 2:]
            mu2 = oriented_points2[j, :2]
            n2 = n2_all[j, :]
            
            # Precomputed covariance and eigenvalues for this correspondence
            sigma2 = sigma2_list[j]
            eigvals2 = eigvals2_list[j]
            p_feat2 = (float(eigvals2[0]), float(eigvals2[1]))
            
            # Compute residual based on cost function (equations 12-14)
            match cost_function:
                case "p2p":
                    residual = point_to_point_cost(mu1, mu2, np.array([[c, -s], [s, c]]), t)
                    # Jacobian for P2P
                    mu1_t = np.array([c * mu1[0] - s * mu1[1], s * mu1[0] + c * mu1[1]]) + t
                    error_vec = mu1_t - mu2
                    error_norm_sq = np.sum(error_vec**2) + 1e-12
                    error_norm = np.sqrt(error_norm_sq)
                    dR_dtheta_mu = np.array([-s * mu1[0] - c * mu1[1], c * mu1[0] - s * mu1[1]])
                    J = np.array([
                        2 * error_vec[0] / error_norm,
                        2 * error_vec[1] / error_norm,
                        2 * float(np.dot(error_vec, dR_dtheta_mu)) / error_norm
                    ], dtype=float)
                case "p2l":
                    residual = point_to_line_cost(mu1, mu2, n2, np.array([[c, -s], [s, c]]), t)
                    # Jacobian for P2L
                    dR_dtheta_mu = np.array([-s * mu1[0] - c * mu1[1], c * mu1[0] - s * mu1[1]])
                    J = np.array([n2[0], n2[1], float(np.dot(n2, dR_dtheta_mu))], dtype=float)
                case "p2d":
                    residual = point_to_distribution_cost(mu1, mu2, sigma2, np.array([[c, -s], [s, c]]), t)
                    # Jacobian for P2D
                    dR_dtheta_mu = np.array([-s * mu1[0] - c * mu1[1], c * mu1[0] - s * mu1[1]])
                    J = np.array([n2[0], n2[1], float(np.dot(n2, dR_dtheta_mu))], dtype=float)
                case _:
                    raise ValueError(f"Unknown cost_function: {cost_function}")

            # Apply Huber loss as weight
            huber_weight = Huber_loss(residual, huber_delta)

            # Compute similarity weights
            eigvals1 = np.linalg.eigvalsh(_covariance_from_normal(n1))
            p_feat1 = (float(eigvals1[0]), float(eigvals1[1]))
            n1_t = np.array([c * n1[0] - s * n1[1], s * n1[0] + c * n1[1]])
            w_similarity = combined_weight(p_feat1, p_feat2, d_count, d_count, n1_t, n2)

            # Combined weight: Huber * similarity
            w = huber_weight * w_similarity

            H += w * np.outer(J, J)
            g += w * J * residual

        last_hessian = H

        # Solve for incremental update
        try:
            delta = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            break

        t += delta[:2]
        theta += delta[2]

        if np.linalg.norm(delta) < 1e-4:
            break

    c = np.cos(theta)
    s = np.sin(theta)
    final_R = np.array([[c, -s], [s, c]], dtype=float)
    
    # Estimate covariance from Hessian: C(xt) = (J^T J)^-1
    if return_covariance:
        try:
            covariance = np.linalg.inv(last_hessian + 1e-9 * np.eye(3))
        except np.linalg.LinAlgError:
            covariance = np.eye(3) * 1e6  # Large uncertainty if singular
        return final_R, t, correspondences, covariance
    
    return final_R, t, correspondences


def _build_cfear_normal_equations(
    source_points,
    target_points,
    theta,
    t,
    cost_function="p2l",
    distance_gate=6.0,
    theta_max_deg=45.0,
    huber_delta=0.5,
):
    source_points = np.asarray(source_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    if source_points.size == 0 or target_points.size == 0:
        return np.zeros((3, 3), dtype=float), np.zeros(3, dtype=float), [], 0.0

    c = np.cos(theta)
    s = np.sin(theta)

    mu1_all = source_points[:, :2]
    n1_all = source_points[:, 2:]
    mu2_all = target_points[:, :2]
    n2_all = target_points[:, 2:]

    mu1_t_all = np.array(
        [
            c * mu1_all[:, 0] - s * mu1_all[:, 1],
            s * mu1_all[:, 0] + c * mu1_all[:, 1],
        ],
        dtype=float,
    ).T + t
    n1_t_all = np.array(
        [
            c * n1_all[:, 0] - s * n1_all[:, 1],
            s * n1_all[:, 0] + c * n1_all[:, 1],
        ],
        dtype=float,
    ).T

    theta_max_rad = np.deg2rad(theta_max_deg)
    cos_theta_max = np.cos(theta_max_rad)
    distance_gate_sq = distance_gate**2

    diff = mu2_all[np.newaxis, :, :] - mu1_t_all[:, np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=2)
    normal_dots = np.dot(n1_t_all, n2_all.T)

    correspondences = []
    for i in range(source_points.shape[0]):
        valid_mask = (dist_sq[i, :] <= distance_gate_sq) & (normal_dots[i, :] >= cos_theta_max)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            best_j = valid_indices[np.argmin(dist_sq[i, valid_indices])]
            correspondences.append((i, best_j))

    if len(correspondences) < 3:
        return np.zeros((3, 3), dtype=float), np.zeros(3, dtype=float), correspondences, 0.0

    sigma2_list = []
    eigvals2_list = []
    for j in range(target_points.shape[0]):
        n2 = target_points[j, 2:]
        sigma2 = _covariance_from_normal(n2)
        sigma2_list.append(sigma2)
        eigvals2_list.append(np.linalg.eigvalsh(sigma2))

    H = np.zeros((3, 3), dtype=float)
    g = np.zeros(3, dtype=float)
    total_cost = 0.0

    d_count = float(len(correspondences))
    R_theta = np.array([[c, -s], [s, c]], dtype=float)
    for i, j in correspondences:
        mu1 = source_points[i, :2]
        n1 = source_points[i, 2:]
        mu2 = target_points[j, :2]
        n2 = n2_all[j, :]

        sigma2 = sigma2_list[j]
        eigvals2 = eigvals2_list[j]
        p_feat2 = (float(eigvals2[0]), float(eigvals2[1]))

        match cost_function:
            case "p2p":
                residual = point_to_point_cost(mu1, mu2, R_theta, t)
                mu1_t = np.array([c * mu1[0] - s * mu1[1], s * mu1[0] + c * mu1[1]]) + t
                error_vec = mu1_t - mu2
                error_norm_sq = np.sum(error_vec**2) + 1e-12
                error_norm = np.sqrt(error_norm_sq)
                dR_dtheta_mu = np.array([-s * mu1[0] - c * mu1[1], c * mu1[0] - s * mu1[1]])
                J = np.array(
                    [
                        2 * error_vec[0] / error_norm,
                        2 * error_vec[1] / error_norm,
                        2 * float(np.dot(error_vec, dR_dtheta_mu)) / error_norm,
                    ],
                    dtype=float,
                )
            case "p2l":
                residual = point_to_line_cost(mu1, mu2, n2, R_theta, t)
                dR_dtheta_mu = np.array([-s * mu1[0] - c * mu1[1], c * mu1[0] - s * mu1[1]])
                J = np.array([n2[0], n2[1], float(np.dot(n2, dR_dtheta_mu))], dtype=float)
            case "p2d":
                residual = point_to_distribution_cost(mu1, mu2, sigma2, R_theta, t)
                dR_dtheta_mu = np.array([-s * mu1[0] - c * mu1[1], c * mu1[0] - s * mu1[1]])
                J = np.array([n2[0], n2[1], float(np.dot(n2, dR_dtheta_mu))], dtype=float)
            case _:
                raise ValueError(f"Unknown cost_function: {cost_function}")

        huber_weight = Huber_loss(residual, huber_delta)
        eigvals1 = np.linalg.eigvalsh(_covariance_from_normal(n1))
        p_feat1 = (float(eigvals1[0]), float(eigvals1[1]))
        n1_t = np.array([c * n1[0] - s * n1[1], s * n1[0] + c * n1[1]])
        w_similarity = combined_weight(p_feat1, p_feat2, d_count, d_count, n1_t, n2)
        w = huber_weight * w_similarity

        H += w * np.outer(J, J)
        g += w * J * residual
        total_cost += w * (residual**2)

    return H, g, correspondences, total_cost


def registration_from_oriented_points(
    oriented_points,
    keyframes=None,
    cost_function="p2l",
    max_iterations=20,
    return_covariance=False,
    start_theta=0.0,
    start_t=np.zeros(2, dtype=float)
):
    """
    Register oriented points.

    Supported modes:
    1) Pair mode (backward compatible):
       - oriented_points=(source, target) and keyframes is None
       - returns transform source -> target
    2) Keyframe mode (joint optimization):
       - oriented_points=source and keyframes=[target_0, target_1, ...]
       - minimizes the summed CFEAR objective over all keyframes.
    """
    if keyframes is None:
        if isinstance(oriented_points, tuple) and len(oriented_points) == 2:
            return _register_pair_cfear(
                oriented_points[0],
                oriented_points[1],
                max_iterations=max_iterations,
                return_covariance=return_covariance,
                cost_function=cost_function,
                start_theta=start_theta,
                start_t=start_t
            )
        if isinstance(oriented_points, list) and len(oriented_points) == 2 and isinstance(oriented_points[0], np.ndarray):
            return _register_pair_cfear(
                oriented_points[0],
                oriented_points[1],
                max_iterations=max_iterations,
                return_covariance=return_covariance,
                cost_function=cost_function,
                start_theta=start_theta,
                start_t=start_t
            )
        raise ValueError("Provide either (source, target) or set keyframes for multi-keyframe registration.")

    source = np.asarray(oriented_points, dtype=float)
    if source.size == 0:
        if return_covariance:
            return np.eye(2), np.zeros(2), [], np.eye(3) * 1e6
        return np.eye(2), np.zeros(2), []

    keyframe_list = [np.asarray(kf, dtype=float) for kf in keyframes if kf is not None and len(kf) > 0]
    if len(keyframe_list) == 0:
        if return_covariance:
            return np.eye(2), np.zeros(2), [], np.eye(3) * 1e6
        return np.eye(2), np.zeros(2), []

    theta = start_theta
    t = np.array(start_t, dtype=float, copy=True)  # Ensure writable copy
    correspondences_all = []
    last_hessian = np.eye(3, dtype=float) * 1e6

    for _iter in range(max_iterations):
        H_total = np.zeros((3, 3), dtype=float)
        g_total = np.zeros(3, dtype=float)
        correspondences_all = []

        for kf_idx, keyframe in enumerate(keyframe_list):
            H_kf, g_kf, corr_kf, _ = _build_cfear_normal_equations(
                source,
                keyframe,
                theta,
                t,
                cost_function=cost_function,
            )
            if len(corr_kf) == 0:
                continue
            H_total += H_kf
            g_total += g_kf
            correspondences_all.extend((kf_idx, i, j) for i, j in corr_kf)

        if len(correspondences_all) < 3:
            break

        last_hessian = H_total
        try:
            delta = -np.linalg.solve(H_total + 1e-9 * np.eye(3), g_total)
        except np.linalg.LinAlgError:
            break

        t += delta[:2]
        theta += delta[2]
        if np.linalg.norm(delta) < 1e-4:
            break

    final_R = _rotation_matrix(theta)
    if return_covariance:
        try:
            covariance = np.linalg.inv(last_hessian + 1e-9 * np.eye(3))
        except np.linalg.LinAlgError:
            covariance = np.eye(3) * 1e6
        return final_R, t, correspondences_all, covariance
    return final_R, t, correspondences_all


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
    
    cartesian_image1, _, _ = polar_to_cartesian_image(img1)
    cartesian_image2, _, _ = polar_to_cartesian_image(img2)
    
    S1, H1 = compute_H_S(img1)
    S2, H2 = compute_H_S(img2)
    
    keypoints1 = Cen2019_keypoints(H1, S1, l_max=100)
    keypoints2 = Cen2019_keypoints(H2, S2, l_max=100)
    
    visualize_keypoints(img1, cartesian_image1, keypoints1)
    visualize_keypoints(img2, cartesian_image2, keypoints2)

    descriptors1 = compute_descriptors(img1, keypoints1)
    descriptors2 = compute_descriptors(img2, keypoints2)

    U = unaryMatchesFromDescriptors(descriptors1, descriptors2)
    print(f"Found {len(U)} unary matches between the two images")
    Compatibility = compute_pairwiseCompatibilityScore(U, keypoints1, keypoints2)
    M = select_matches(U, Compatibility)

    visualize_matches(img1, img2, keypoints1, keypoints2, M)
