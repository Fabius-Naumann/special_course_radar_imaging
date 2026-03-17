import numpy as np
from sklearn.linear_model import RANSACRegressor

from utils.data_loading import polar_to_cartesian_points

# RANSAC
def motion_estimation_ransac(matches, kp1, kp2):
    """
    Estimate the motion (bearing and translation) between two sets of keypoints using RANSAC.
    
    Parameters:
    - matches: List of matched keypoint indices (from data association)
    - kp1: Keypoints from image 1 (Nx2 array of (angle, range))
    - kp2: Keypoints from image 2 (Nx2 array of (angle, range))
    
    Returns:
    - theta_deg: Motion bearing in degrees
    - best_t: Estimated translation vector
    - points1: Cartesian coordinates of matched points from image 1
    - points2: Cartesian coordinates of matched points from image 2
    - best_inliers: Boolean mask indicating inlier points
    """
    # Extract matched keypoints
    matched_kp1 = np.array([kp1[int(m[0])] for m in matches])
    matched_kp2 = np.array([kp2[int(m[1])] for m in matches])

    # polar_to_cartesian_points handles resolution scaling internally
    points1 = polar_to_cartesian_points(matched_kp1[:, 1], matched_kp1[:, 0])  # Convert (angle, range) to (x, y)
    points2 = polar_to_cartesian_points(matched_kp2[:, 1], matched_kp2[:, 0])  # Convert (angle, range) to (x, y)
    
    # Use RANSAC to estimate the best transformation
    ransac = RANSACRegressor(max_trials=1000)
    ransac.fit(points1, points2)
    best_R = ransac.estimator_.coef_
    best_t = ransac.estimator_.intercept_
    best_inliers = ransac.inlier_mask_

    # Extract motion bearing from transformation matrix
    theta_rad = np.arctan2(best_R[1, 0], best_R[0, 0])
    theta_deg = np.degrees(theta_rad) % 360

    return theta_deg, best_t, points1, points2, best_inliers

# SVD
def motion_estimation_SVD(matches, kp1, kp2):
    """
    Estimate the motion (bearing and translation) between two sets of keypoints using SVD.
    
    Parameters:
    - matches: List of matched keypoint indices (from data association)
    - kp1: Keypoints from image 1 (Nx2 array of (angle, range))
    - kp2: Keypoints from image 2 (Nx2 array of (angle, range))
    
    Returns:
    - theta_deg: Motion bearing in degrees
    - best_t: Estimated translation vector
    """
    # Extract matched keypoints
    matched_kp1 = np.array([kp1[int(m[0])] for m in matches])
    matched_kp2 = np.array([kp2[int(m[1])] for m in matches])

    # polar_to_cartesian_points handles resolution scaling internally
    points1 = polar_to_cartesian_points(matched_kp1[:, 1], matched_kp1[:, 0])  # Convert (angle, range) to (x, y)
    points2 = polar_to_cartesian_points(matched_kp2[:, 1], matched_kp2[:, 0])  # Convert (angle, range) to (x, y)

    # Compute centroids
    centroid_1 = np.mean(points1, axis=0)
    centroid_2 = np.mean(points2, axis=0)
    
    # Center the points
    centered_1 = points1 - centroid_1
    centered_2 = points2 - centroid_2

    # Compute covariance matrix
    n = points1.shape[0]
    C = 1/n * centered_1.T @ centered_2

    # SVD decomposition
    U, _, Vt = np.linalg.svd(C)

    det = np.linalg.det(Vt.T @ U.T)
    
    # Compute rotation matrix (Kabsch algorithm: R = V @ D @ U^T)
    R = Vt.T @ np.array([[1, 0], [0, det]]) @ U.T

    # Compute translation
    t = centroid_2 - R @ centroid_1

    return R, t


# GNC

# Windowed Bundle Adjustment
def motion_estimation_bundle_adjustment(matches, kp, prev_trans, prev_rot, method = "RANSAC"):
    """
    Refine motion estimation using bundle adjustment.
    
    Parameters:
    - matches: List of matched keypoint indices for the current image with n previous images (from data association)
    - kp: (n+1) Keypoints from the current image and the n previous images (Nx2 array of (angle, range))
    - method: motion estimation method ("RANSAC" or "SVD")
    - max_iterations: Maximum number of optimization iterations
    - tolerance: Convergence threshold for optimization
    
    Returns:
    - refined_R: Refined rotation matrix
    - refined_t: Refined translation vector
    """
    n = len(matches)  # Number of previous images

    # Initialization
    kp1 = kp[0]  # Keypoints from the current image
    kp_prev = kp[1:]  # Keypoints from the n previous images

    if method == "RANSAC":
        initial_theta, initial_t, _, _, _ = motion_estimation_ransac(matches[0], kp1, kp_prev[0])
        theta = np.radians(float(initial_theta))
        t = np.array(initial_t, dtype=float).ravel().copy()
    else:  # SVD
        initial_R, initial_t = motion_estimation_SVD(matches[0], kp1, kp_prev[0])

    # Windowed Bundle Adjustment
    pass
    

    