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

    if points1.size == 0 or points2.size == 0:
        raise ValueError("motion_estimation_SVD requires at least one matched point")

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

    

    