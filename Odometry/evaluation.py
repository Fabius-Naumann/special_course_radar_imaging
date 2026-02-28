import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import RANSACRegressor
from geopy.distance import geodesic

from data_loading import polar_to_cartesian_points, extract_timestamp, load_gps_data, load_radar_images, polar_to_cartesian_image
from keypoint_extraction import compute_H_S, extract_keypoints
from data_association import compute_descriptors, unaryMatchesFromDescriptors, compute_pairwiseCompatibilityScore, select_matches

def motion_estimation(matches, kp1, kp2):
    """
    Estimate the motion (rotation and translation) between two sets of keypoints using RANSAC.
    
    Parameters:
    - matches: List of matched keypoint indices (from data association)
    - kp1: Keypoints from image 1 (Nx2 array of (angle, range))
    - kp2: Keypoints from image 2 (Nx2 array of (angle, range))
    
    Returns:
    - R: Estimated rotation matrix
    - t: Estimated translation vector
    """
    # Extract matched keypoints
    matched_kp1 = np.array([kp1[int(m[0])] for m in matches])
    matched_kp2 = np.array([kp2[int(m[1])] for m in matches])

    #print(f"Matched keypoints in polar coordinates:\nImage 1:\n{matched_kp1}\nImage 2:\n{matched_kp2}")
    
    # Transform using resolution to meters and degrees
    range_resolution = 1000/6448
    angle_resolution = 360/400

    matched_kp1 = np.column_stack((matched_kp1[:, 0] * angle_resolution, matched_kp1[:, 1] * range_resolution))
    matched_kp2 = np.column_stack((matched_kp2[:, 0] * angle_resolution, matched_kp2[:, 1] * range_resolution))

    points1 = polar_to_cartesian_points(matched_kp1[:, 1], matched_kp1[:, 0])  # Convert (angle, range) to (x, y)
    points2 = polar_to_cartesian_points(matched_kp2[:, 1], matched_kp2[:, 0])  # Convert (angle, range) to (x, y)
    
    # Use RANSAC to estimate the best transformation
    ransac = RANSACRegressor(max_trials=1000)
    ransac.fit(points1, points2)
    best_R = ransac.estimator_.coef_
    best_t = ransac.estimator_.intercept_
    best_inliers = ransac.inlier_mask_

    # transfrom in rotation angle
    theta_rad = np.arctan2(best_R[1, 0], best_R[0, 0])
    theta_deg = np.degrees(theta_rad)

    return theta_deg, best_t

def extract_timeframe(gps_data, timestamps, time_tolerance=0.5):
    """
    Extract GPS data for the timeframe between two timestamps.
    
    Parameters:
    - gps_data: DataFrame containing GPS data with a 'timestamp' column
    - timestamps: List of timestamps for radar images
    - time_tolerance: Time tolerance in seconds
    
    Returns:
    - gps_subset: DataFrame containing GPS data within the specified timeframe
    """
    gps_subset = pd.DataFrame()

    for t in timestamps:
        # Convert timestamps to datetime objects
        radar_time_point = pd.to_datetime(t)

        # Create time window
        time_window_start = radar_time_point - timedelta(seconds=time_tolerance)
        time_window_end = radar_time_point #+ timedelta(seconds=time_tolerance)

        # Extract GPS data within the time window
        subset = gps_data[(gps_data['time-string'] >= time_window_start) & (gps_data['time-string'] <= time_window_end)]
        if subset.empty:
            print(f"No GPS data found for timestamp {t} within the time window {time_window_start} - {time_window_end}.")
        else:
            gps_subset = pd.concat([gps_subset, subset], ignore_index=True)

    gps_subset = gps_subset.drop_duplicates().reset_index(drop=True)
    return gps_subset    
    

def calculate_gps_distance(gps_entry1, gps_entry2):
    """
    Calculate the geodesic distance between two GPS entries.
    
    Parameters:
    - gps_entry1: First GPS entry with 'latitude' and 'longitude' fields
    - gps_entry2: Second GPS entry with 'latitude' and 'longitude' fields
    
    Returns:
    - distance: Geodesic distance in meters
    """
    coord1 = (gps_entry1['latitude'], gps_entry1['longitude'])
    coord2 = (gps_entry2['latitude'], gps_entry2['longitude'])
    
    distance = geodesic(coord1, coord2).meters
    return distance

def calculate_gps_bearing(gps_entry1, gps_entry2):
    """
    Calculate the bearing between two GPS entries.
    
    Parameters:
    - gps_entry1: First GPS entry with 'latitude' and 'longitude' fields
    - gps_entry2: Second GPS entry with 'latitude' and 'longitude' fields
    
    Returns:
    - bearing: Bearing in degrees
    """
    lat1 = np.radians(gps_entry1['latitude'])
    lat2 = np.radians(gps_entry2['latitude'])
    diff_long = np.radians(gps_entry2['longitude'] - gps_entry1['longitude'])

    x = np.sin(diff_long) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))

    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def calculate_gt_motion(gps_subset):
    """
    Calculate the ground truth motion (translation) from GPS data.
    
    Parameters:
    - gps_subset: DataFrame containing GPS data for the relevant timeframe
    
    Returns:
    - gt_translation: Ground truth translation in meters
    - gt_bearing: Ground truth bearing in degrees
    """
    if len(gps_subset) < 2:
        print("Not enough GPS data to calculate ground truth motion.")
        return None
    
    gt_translation = []
    gt_bearing = []

    for i in range(len(gps_subset) - 1):
        gps_entry1 = gps_subset.iloc[i]
        gps_entry2 = gps_subset.iloc[i + 1]
        distance = calculate_gps_distance(gps_entry1, gps_entry2)
        gt_translation.append(distance)
        bearing = calculate_gps_bearing(gps_entry1, gps_entry2)
        gt_bearing.append(bearing)

    return gt_translation, gt_bearing


if __name__ == "__main__":
    image_file, image = load_radar_images(num_images=2)
    img1 = image[0]
    img2 = image[1]
    
    timestamp1 = extract_timestamp(image_file[0])
    timestamp2 = extract_timestamp(image_file[1])
    
    cartesian_image1 = polar_to_cartesian_image(img1)
    cartesian_image2 = polar_to_cartesian_image(img2)
    
    S1, H1 = compute_H_S(img1)
    S2, H2 = compute_H_S(img2)
    
    keypoints1 = extract_keypoints(H1, S1, l_max=100)
    keypoints2 = extract_keypoints(H2, S2, l_max=100)

    descriptors1 = compute_descriptors(img1, keypoints1)
    descriptors2 = compute_descriptors(img2, keypoints2)

    U = unaryMatchesFromDescriptors(descriptors1, descriptors2)
    print(f"Found {len(U)} unary matches between the two images")
    Compatibility = compute_pairwiseCompatibilityScore(U, keypoints1, keypoints2)
    M = select_matches(U, Compatibility)

    r, t = motion_estimation(M, keypoints1, keypoints2)
    print(f"Estimated rotation:\n{r}°\nEstimated translation:\n{t} m")

    gps_data = load_gps_data()
    gps_subset = extract_timeframe(gps_data, [timestamp1, timestamp2])
    gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
    print(f"Ground truth translation from GPS data: {gt_translation} m")
    print(f"Ground truth bearing from GPS data: {gt_bearing} degrees")
