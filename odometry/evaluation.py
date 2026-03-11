import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import RANSACRegressor
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import folium
from scipy.ndimage import shift

from utils.data_loading import polar_to_cartesian_points, extract_timestamp, load_gps_data, load_radar_images, polar_to_cartesian_image
from keypoint_extraction import compute_H_S, extract_keypoints
from data_association import compute_descriptors, unaryMatchesFromDescriptors, compute_pairwiseCompatibilityScore, select_matches

def motion_estimation(matches, kp1, kp2):
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

    det = np.linalg.det(U @ Vt)
    
    # Compute rotation matrix
    R = U @ [[1, 0], [0, det]] @ Vt.T 

    # Compute translation
    t = centroid_2 - R @ centroid_1

    return R, t

def visualize_ransac_motion(points1, points2, inliers, best_R, best_t, theta_deg):
    """
    Visualize RANSAC motion estimation results.
    
    Parameters:
    - points1: Cartesian points from image 1 (Nx2)
    - points2: Cartesian points from image 2 (Nx2)
    - inliers: Boolean mask indicating inlier points
    - best_R: Estimated transformation matrix (2x2)
    - best_t: Estimated translation vector
    - theta_deg: Motion bearing in degrees
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Inliers vs Outliers
    ax1 = axes[0]
    inlier_indices = np.where(inliers)[0]
    outlier_indices = np.where(~inliers)[0]
    
    # Plot matched points
    ax1.scatter(points1[:, 0], points1[:, 1], c='blue', s=50, alpha=0.6, label='Points from Image 1')
    ax1.scatter(points2[:, 0], points2[:, 1], c='green', s=50, alpha=0.6, label='Points from Image 2')
    
    # Highlight inliers and outliers
    if len(inlier_indices) > 0:
        ax1.scatter(points1[inlier_indices, 0], points1[inlier_indices, 1], 
                   c='red', s=100, marker='o', alpha=0.9, label='Inliers', edgecolors='darkred', linewidth=2)
    if len(outlier_indices) > 0:
        ax1.scatter(points1[outlier_indices, 0], points1[outlier_indices, 1], 
                   c='orange', s=100, marker='x', alpha=0.9, label='Outliers', linewidth=2)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('RANSAC: Inliers vs Outliers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Transformation visualization
    ax2 = axes[1]
    
    # Transform points1 using the estimated transformation
    points1_transformed = (points1 @ best_R.T) + best_t
    
    # Plot original and transformed points
    ax2.scatter(points1[:, 0], points1[:, 1], c='blue', s=50, alpha=0.6, label='Points from Image 1')
    ax2.scatter(points1_transformed[:, 0], points1_transformed[:, 1], c='red', s=50, alpha=0.6, label='Transformed Points')
    ax2.scatter(points2[:, 0], points2[:, 1], c='green', s=50, alpha=0.6, label='Points from Image 2')
    
    # Draw arrows showing transformation for inliers
    if len(inlier_indices) > 0:
        for idx in inlier_indices:  # Show arrows for all inliers to avoid clutter
            ax2.arrow(points1[idx, 0], points1[idx, 1], 
                     points1_transformed[idx, 0] - points1[idx, 0], 
                     points1_transformed[idx, 1] - points1[idx, 1],
                     head_width=0.02, head_length=0.02, fc='purple', ec='purple', alpha=0.5)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Motion Estimation (Bearing: {theta_deg:.2f}°, Translation: [{best_t[0]:.3f}, {best_t[1]:.3f}] m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Add statistics text
    num_inliers = np.sum(inliers)
    num_outliers = np.sum(~inliers)
    stats_text = f"Inliers: {num_inliers}\nOutliers: {num_outliers}\nInlier ratio: {num_inliers/(num_inliers+num_outliers)*100:.1f}%"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

def visualize_transformed_overlay(img1, img2, theta_deg, translation):
    """
    Visualize the transformed first image next to the second image for comparison.
    
    Parameters:
    - img1: First radar image (polar coordinates)
    - img2: Second radar image (polar coordinates)
    - theta_deg: Motion bearing in degrees (not image rotation)
    - translation: Estimated translation vector [tx, ty] in meters
    """
    # Convert images to Cartesian coordinates
    cart_img1 = polar_to_cartesian_image(img1)
    cart_img2 = polar_to_cartesian_image(img2)
    
    # Convert translation from meters to pixels
    # Estimate pixel scaling (this is approximate based on image size and range)
    max_range_m = 1000  # meters
    image_radius_pixels = min(cart_img1.shape) / 2
    pixels_per_meter = image_radius_pixels / max_range_m
    
    translation_pixels = translation * pixels_per_meter
    
    # Apply translation only (no rotation - theta is motion bearing, not image rotation)
    cart_img1_transformed = shift(cart_img1, 
                                   [translation_pixels[1], translation_pixels[0]], 
                                   mode='constant', cval=0)
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Transformed Image 1
    ax1 = axes[0]
    im1 = ax1.imshow(cart_img1_transformed, cmap='hot', origin='lower')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title(f'Image 1 (Transformed)\nMotion Bearing: {theta_deg:.2f}°\nTranslation: [{translation[0]:.2f}, {translation[1]:.2f}] m')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot 2: Image 2 (Reference)
    ax2 = axes[1]
    im2 = ax2.imshow(cart_img2, cmap='hot', origin='lower')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_title('Image 2 (Reference)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Plot 3: Overlay for comparison
    ax3 = axes[2]
    # Create RGB overlay: Image 1 in red channel, Image 2 in green channel
    # Normalize images to [0, 1] range
    img1_norm = (cart_img1_transformed - cart_img1_transformed.min()) / (cart_img1_transformed.max() - cart_img1_transformed.min() + 1e-8)
    img2_norm = (cart_img2 - cart_img2.min()) / (cart_img2.max() - cart_img2.min() + 1e-8)
    
    # Create RGB overlay
    overlay = np.zeros((*cart_img2.shape, 3))
    overlay[:, :, 0] = img1_norm  # Red channel: transformed image 1
    overlay[:, :, 1] = img2_norm  # Green channel: image 2
    # Where they overlap well, you'll see yellow
    
    ax3.imshow(overlay, origin='lower')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.set_title('Overlay (Red: Img1, Green: Img2, Yellow: Match)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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
        return None, None
    
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

def map_gps(gps_data, odo_trans, timestamps, output_file="odometry\gps_map.html"):
    """
    Visualize GPS points and trajectory on an OpenStreetMap basemap.
    
    Parameters:
    - gps_data: Full DataFrame containing GPS data with 'latitude', 'longitude', and 'time-string' fields
    - odo_trans: List of odometry translations 
    - timestamps: List of timestamps to extract GPS data for
    - output_file: HTML file path for the generated interactive map
    """
    # Extract GPS data for the relevant timeframe
    gps_subset = extract_timeframe(gps_data, timestamps)
    
    if gps_subset.empty:
        print("No GPS data to visualize on the map.")
        return

    gps_plot = gps_subset.copy().reset_index(drop=True)
    coordinates = gps_plot[['latitude', 'longitude']].values.tolist()

    # Build odometry trajectory by converting translations to GPS coordinates
    # Start from first GPS coordinate
    coordinates_odo = [coordinates[0]]
    current_lat, current_lon = coordinates[0]
    
    # Convert odometry translations (in meters) to GPS coordinate offsets
    for trans in odo_trans:
        # Handle both numpy arrays and regular lists
        if hasattr(trans, '__iter__') and len(trans) >= 2:
            dx, dy = float(trans[0]), float(trans[1])
            
            # Approximate conversion: 1 degree latitude ≈ 111,320 meters
            # 1 degree longitude varies with latitude: ≈ 111,320 * cos(latitude)
            lat_offset = dy / 111320.0
            lon_offset = dx / (111320.0 * np.cos(np.radians(current_lat)))
            
            current_lat += lat_offset
            current_lon += lon_offset
            coordinates_odo.append([current_lat, current_lon])
        else:
            # If trans is malformed, skip it
            continue

    center_lat = float(gps_plot['latitude'].mean())
    center_lon = float(gps_plot['longitude'].mean())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    # GPS trajectory in red
    if len(coordinates) > 1:
        folium.PolyLine(coordinates, color="red", weight=3, opacity=0.8).add_to(fmap)

    for i, (lat, lon) in enumerate(coordinates):
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="red",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Point {i}"
        ).add_to(fmap)

    if len(coordinates_odo) > 1:
        folium.PolyLine(coordinates_odo, color="green", weight=3, opacity=0.8).add_to(fmap)

    # Odometry trajectory in green
    for i, (lat, lon) in enumerate(coordinates_odo):
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="green",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Odometry Point {i}"
        ).add_to(fmap)

    if len(coordinates) > 0:
        folium.Marker(coordinates[0], tooltip="Start").add_to(fmap)
    if len(coordinates) > 1:
        folium.Marker(coordinates[-1], tooltip="End").add_to(fmap)
    if len(coordinates_odo) > 1:
        folium.Marker(coordinates_odo[-1], tooltip="Odometry End").add_to(fmap)

    fmap.save(output_file)
    print(f"Saved OpenStreetMap visualization to {output_file}")

if __name__ == "__main__":
    image_file, image = load_radar_images(num_images=5)
    img1 = image[2]
    img2 = image[3]
    
    timestamp1 = extract_timestamp(image_file[0])
    timestamp2 = extract_timestamp(image_file[1])
    print(f"Timestamp for Image 1: {timestamp1}")
    print(f"Timestamp for Image 2: {timestamp2}")
    
    cartesian_image1 = polar_to_cartesian_image(img1)
    cartesian_image2 = polar_to_cartesian_image(img2)
    
    S1, H1 = compute_H_S(img1)
    S2, H2 = compute_H_S(img2)
    
    keypoints1 = extract_keypoints(H1, S1, l_max=200)
    keypoints2 = extract_keypoints(H2, S2, l_max=200)

    descriptors1 = compute_descriptors(img1, keypoints1)
    descriptors2 = compute_descriptors(img2, keypoints2)

    U = unaryMatchesFromDescriptors(descriptors1, descriptors2)
    print(f"Found {len(U)} unary matches between the two images")
    Compatibility = compute_pairwiseCompatibilityScore(U, keypoints1, keypoints2)
    M = select_matches(U, Compatibility)

    theta_deg, t, points1, points2, inliers = motion_estimation(M, keypoints1, keypoints2)
    print(f"Motion bearing:\n{theta_deg}°\nEstimated translation:\n{t} m")
    
    # Reconstruct best_R for visualization
    best_R = np.array([[np.cos(np.radians(theta_deg)), -np.sin(np.radians(theta_deg))],
                       [np.sin(np.radians(theta_deg)), np.cos(np.radians(theta_deg))]])
    
    # Visualize RANSAC results
    visualize_ransac_motion(points1, points2, inliers, best_R, t, theta_deg)
    
    # Visualize transformed image comparison
    visualize_transformed_overlay(img1, img2, theta_deg, t)

    gps_data = load_gps_data()
    gps_subset = extract_timeframe(gps_data, [timestamp1, timestamp2])
    gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
    print(f"Ground truth translation from GPS data: {gt_translation} m")
    print(f"Ground truth bearing from GPS data: {gt_bearing} degrees")
    map_gps(gps_data, [timestamp1, timestamp2])
