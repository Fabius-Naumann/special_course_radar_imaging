import numpy as np
import pandas as pd
import os
import cv2
from datetime import timedelta
from sklearn.linear_model import RANSACRegressor
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import folium
from scipy.ndimage import shift, rotate

import sys
from pathlib import Path

# Add the project root folder to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loading import extract_gps_timeframe, extract_timestamp, load_gps_data, load_radar_images, polar_to_cartesian_image
from keypoint_extraction import Cen2019_keypoints, compute_H_S
from data_association import compute_descriptors, unaryMatchesFromDescriptors, compute_pairwiseCompatibilityScore, select_matches
from motion_estimation import motion_estimation_ransac, motion_estimation_SVD

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
    cart_img1, _, _ = polar_to_cartesian_image(img1)
    cart_img2, _, _ = polar_to_cartesian_image(img2)
    
    # Convert translation from meters to pixels
    # Estimate pixel scaling (this is approximate based on image size and range)
    max_range_m = 1000  # meters
    image_radius_pixels = min(cart_img1.shape) / 2
    pixels_per_meter = image_radius_pixels / max_range_m
    
    translation_pixels = translation * pixels_per_meter
    
    # Apply rotation first, then translation
    cart_img1_rotated = rotate(cart_img1, theta_deg, reshape=False, mode='constant', cval=0)
    cart_img1_transformed = shift(cart_img1_rotated,
                                   [translation_pixels[1], translation_pixels[0]],
                                   mode='constant', cval=0)

    # Normalize images to [0, 1] range
    img1_norm = (cart_img1_transformed - cart_img1_transformed.min()) / (cart_img1_transformed.max() - cart_img1_transformed.min() + 1e-8)
    img2_norm = (cart_img2 - cart_img2.min()) / (cart_img2.max() - cart_img2.min() + 1e-8)

    # Create RGB overlay: Image 1 in red channel, Image 2 in green channel
    # Where they overlap well, you'll see yellow
    overlay = np.zeros((*cart_img2.shape, 3))
    overlay[:, :, 0] = img1_norm  # Red channel: transformed image 1
    overlay[:, :, 1] = img2_norm  # Green channel: image 2

    overlay_zoomed = overlay[300:700, 300:700]

    # Create 2x2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Transformed Image 1 (top-left)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(cart_img1_transformed, cmap='hot', origin='lower')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title(f'Image 1 (Transformed)\nRotation: {theta_deg:.2f}°, Translation: [{translation[0]:.2f}, {translation[1]:.2f}] m')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot 2: Image 2 (Reference, top-right)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(cart_img2, cmap='hot', origin='lower')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_title('Image 2 (Reference)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Plot 3: Full Overlay (bottom-left)
    ax3 = axes[1, 0]
    ax3.imshow(overlay, origin='lower')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.set_title('Overlay (Red: Img1, Green: Img2, Yellow: Match)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Zoomed Overlay (bottom-right)
    ax4 = axes[1, 1]
    ax4.imshow(overlay_zoomed, origin='lower')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    ax4.set_title('Zoomed Overlay (Red: Img1, Green: Img2, Yellow: Match)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#TODO: not working yet
def create_stabilized_overlay_video(
    images,
    rotations_deg,
    translations,
    output_file="odometry/transformed_overlay.mp4",
    fps: int = 8,
    frames_per_image: int = 4,
):
    """
    Create a video where each radar frame is transformed into the first-frame
    coordinate system and shown directly (no overlay).

    Parameters:
    - images: list/array of radar images in polar coordinates
        - rotations_deg: list of incremental rotations (len = N-1), each for
            transform from frame (i+1) into frame i
        - translations: list of incremental translations [tx, ty] in meters (len = N-1),
            each for transform from frame (i+1) into frame i
    - output_file: target mp4 file
    - fps: output video frame rate
    - frames_per_image: number of repeated video frames per radar image
    """
    if images is None or len(images) == 0:
        raise ValueError("'images' must contain at least one frame.")

    n_frames = len(images)
    # Accept either incremental transforms (len = n_frames-1) or absolute poses (len = n_frames).
    # If absolute poses are provided, convert to incremental steps internally.
    rot_arr = np.asarray(rotations_deg, dtype=float).reshape(-1)
    trans_arr = np.asarray(translations, dtype=float)
    if trans_arr.ndim == 1:
        trans_arr = trans_arr.reshape(-1, 2)

    def shortest_signed_angle_diff_deg(a):
        d = np.diff(a)
        d = (d + 180.0) % 360.0 - 180.0
        return d

    if len(rot_arr) == n_frames:
        rotations_inc_deg = shortest_signed_angle_diff_deg(rot_arr)
    elif len(rot_arr) == n_frames - 1:
        rotations_inc_deg = rot_arr
    elif len(rot_arr) == 0:
        rotations_inc_deg = np.empty((0,), dtype=float)
    else:
        raise ValueError("'rotations_deg' must have length N or N-1 where N=len(images)")

    if trans_arr.shape[0] == n_frames:
        translations_inc_xy = np.diff(trans_arr, axis=0)
    elif trans_arr.shape[0] == n_frames - 1:
        translations_inc_xy = trans_arr
    elif trans_arr.shape[0] == 0:
        translations_inc_xy = np.empty((0, 2), dtype=float)
    else:
        raise ValueError("'translations' must have shape (N,2) or (N-1,2) where N=len(images)")
    
    if int(frames_per_image) < 1:
        raise ValueError("'frames_per_image' must be >= 1.")

    cart_images = [polar_to_cartesian_image(img)[0].astype(np.float32) for img in images]
    h, w = cart_images[0].shape

    # Approximate radar image metric scale (same assumption as visualization code).
    max_range_m = 1000.0
    image_radius_pixels = min(h, w) / 2.0
    pixels_per_meter = image_radius_pixels / max_range_m

    ref = cart_images[0]

    def normalize_to_uint8(arr):
        arr = np.asarray(arr, dtype=np.float32)
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        if arr_max - arr_min < 1e-8:
            return np.zeros_like(arr, dtype=np.uint8)
        arr_norm = (arr - arr_min) / (arr_max - arr_min)
        return np.clip(arr_norm * 255.0, 0, 255).astype(np.uint8)

    ref_u8 = normalize_to_uint8(ref)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for '{output_file}'.")

    try:
        # Frame 0: reference image.
        first_frame_bgr = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
        for _ in range(frames_per_image):
            writer.write(first_frame_bgr)

        # Cumulative transform mapping frame i -> frame 0.
        # With incremental transforms T_(i-1)<-i, composition is:
        # T_0<-i = T_0<-(i-1) @ T_(i-1)<-i
        T_0_from_i = np.eye(3, dtype=np.float64)

        # Pose transforms are defined around radar center (origin at image center),
        # while OpenCV affine warps use top-left pixel origin. Convert by conjugation:
        # T_pixel = C @ T_center @ C^{-1}
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        C = np.array(
            [[1.0, 0.0, cx],
             [0.0, 1.0, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        C_inv = np.array(
            [[1.0, 0.0, -cx],
             [0.0, 1.0, -cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        for i in range(1, n_frames):
            # Odometry is in math coordinates (x right, y up, CCW positive).
            # Images are in pixel coordinates (x right, y down), so convert via:
            # theta_img = -theta_math, t_img = [tx, -ty].
            theta = np.radians(-float(rotations_inc_deg[i - 1]))
            c, s = np.cos(theta), np.sin(theta)
            tx_m, ty_m = np.asarray(translations_inc_xy[i - 1], dtype=float)[:2]
            tx_px = tx_m * pixels_per_meter
            ty_px = -ty_m * pixels_per_meter

            T_prev_from_curr = np.array(
                [[c, -s, tx_px],
                 [s,  c, ty_px],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

            T_0_from_i = T_0_from_i @ T_prev_from_curr

            # Warp frame i into frame-0 coordinates using source->destination matrix,
            # converted from center-origin to pixel-origin coordinates.
            T_pixel = C @ T_0_from_i @ C_inv
            M = T_pixel[:2, :]
            warped = cv2.warpAffine(
                cart_images[i],
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            warped_u8 = normalize_to_uint8(warped)
            frame_bgr = cv2.cvtColor(warped_u8, cv2.COLOR_GRAY2BGR)
            for _ in range(frames_per_image):
                writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"Saved transformed overlay video to {output_file}")

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
    gps_subset = extract_gps_timeframe(gps_data, timestamps, time_tolerance=time_tolerance)
    if gps_subset.empty:
        for timestamp in timestamps:
            radar_time_point = pd.to_datetime(timestamp)
            time_window_start = radar_time_point - timedelta(seconds=time_tolerance)
            print(
                f"No GPS data found for timestamp {timestamp} "
                f"within the time window {time_window_start} - {radar_time_point}."
            )
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

def extract_gps_bearing(gps_subset):
    """
    Extract the bearing from GPS data.
    
    Parameters:
    - gps_subset: DataFrame containing GPS data with a 'bearing' column
    
    Returns:
    - bearing: Bearing in degrees (mean of available bearings)
    """
    if 'bearing' in gps_subset.columns and not gps_subset['bearing'].isnull().all():
        return gps_subset['bearing'].mean()
    else:
        print("No valid 'bearing' data found in GPS subset.")
        return None

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
    #gt_bearing = []

    for i in range(len(gps_subset) - 1):
        gps_entry1 = gps_subset.iloc[i]
        gps_entry2 = gps_subset.iloc[i + 1]
        distance = calculate_gps_distance(gps_entry1, gps_entry2)
        gt_translation.append(distance)
        #bearing = calculate_gps_bearing(gps_entry1, gps_entry2)
        #gt_bearing.append(bearing)

    return gt_translation #, gt_bearing

def interpolate_gps_motion(gps_subset, timestamps):
    """
    Interpolate GPS motion to estimate position at specific timestamps.
    
    Parameters:
    - gps_subset: DataFrame containing GPS data for the relevant timeframe
    - timestamps: List of timestamps to interpolate GPS positions for
    
    Returns:
    - interpolated_positions: DataFrame with columns ['time-string', 'latitude', 'longitude', 'bearing']
    """
    interpolated_positions = []
    gps_sorted = gps_subset.sort_values('time-string').reset_index(drop=True)

    def interpolate_bearing(before_bearing, after_bearing, ratio):
        """Interpolate compass angles (degrees) along the shortest arc."""
        if pd.isna(before_bearing) and pd.isna(after_bearing):
            return np.nan
        if pd.isna(before_bearing):
            return float(after_bearing)
        if pd.isna(after_bearing):
            return float(before_bearing)

        before_bearing = float(before_bearing) % 360.0
        after_bearing = float(after_bearing) % 360.0
        delta = ((after_bearing - before_bearing + 180.0) % 360.0) - 180.0
        return (before_bearing + ratio * delta) % 360.0
    
    for t in timestamps:
        radar_time_point = pd.to_datetime(t)
        before_candidates = gps_sorted[gps_sorted['time-string'] <= radar_time_point]
        after_candidates = gps_sorted[gps_sorted['time-string'] >= radar_time_point]

        # Clamp to nearest available GPS point if timestamp is outside GPS range
        if before_candidates.empty:
            nearest = after_candidates.iloc[0]
            interpolated_positions.append({
                'time-string': radar_time_point,
                'latitude': nearest['latitude'],
                'longitude': nearest['longitude'],
                'bearing': nearest['bearing'] if 'bearing' in gps_sorted.columns else np.nan
            })
            continue

        if after_candidates.empty:
            nearest = before_candidates.iloc[-1]
            interpolated_positions.append({
                'time-string': radar_time_point,
                'latitude': nearest['latitude'],
                'longitude': nearest['longitude'],
                'bearing': nearest['bearing'] if 'bearing' in gps_sorted.columns else np.nan
            })
            continue

        # Find the two GPS entries surrounding the radar timestamp
        before = before_candidates.iloc[-1]
        after = after_candidates.iloc[0]
        
        if before['time-string'] == after['time-string']:
            # If the radar timestamp exactly matches a GPS entry, use that position
            interpolated_positions.append({
                'time-string': radar_time_point,
                'latitude': before['latitude'],
                'longitude': before['longitude'],
                'bearing': before['bearing'] if 'bearing' in gps_sorted.columns else np.nan
            })
        else:
            # Linear interpolation based on time
            total_time_diff = (after['time-string'] - before['time-string']).total_seconds()
            if total_time_diff == 0:
                interpolated_positions.append({
                    'time-string': radar_time_point,
                    'latitude': before['latitude'],
                    'longitude': before['longitude'],
                    'bearing': before['bearing'] if 'bearing' in gps_sorted.columns else np.nan
                })
                continue
            
            time_diff_to_before = (radar_time_point - before['time-string']).total_seconds()
            ratio = time_diff_to_before / total_time_diff
            
            lat_interp = before['latitude'] + ratio * (after['latitude'] - before['latitude'])
            lon_interp = before['longitude'] + ratio * (after['longitude'] - before['longitude'])
            bearing_interp = (
                interpolate_bearing(before['bearing'], after['bearing'], ratio)
                if 'bearing' in gps_sorted.columns
                else np.nan
            )
            
            interpolated_positions.append({
                'time-string': radar_time_point,
                'latitude': lat_interp,
                'longitude': lon_interp,
                'bearing': bearing_interp
            })
    
    return pd.DataFrame(interpolated_positions)

def imu_to_xytheta(imu_data):
    """
    imu_data: pandas DataFrame with columns angular_velocity_x	angular_velocity_y	angular_velocity_z	linear_acceleration_x	linear_acceleration_y	linear_acceleration_z

    Returns:
        dx:   N array
        dy:   N array
        dtheta: N array
    """
    # Time differences between timestamps, expressed in seconds.
    timestamps = pd.to_datetime(imu_data['timestamp'], errors='coerce').to_numpy(dtype='datetime64[ns]')
    dt = np.diff(timestamps).astype('timedelta64[ns]').astype(np.float64) / 1e9
    dt = np.append(dt, dt[-1])  # Assume constant dt for the last element

    # Extract body-frame acceleration and yaw rate
    ax = imu_data['linear_acceleration_x'].values
    ay = imu_data['linear_acceleration_y'].values
    wz = imu_data['angular_velocity_z'].values   # yaw rate

    # Integrate yaw rate to get heading increments
    dtheta = wz * dt

    # Rotate body-frame acceleration into world frame using incremental heading
    theta = np.cumsum(dtheta)  # heading over time

    # World-frame acceleration
    ax_w = ax * np.cos(theta) - ay * np.sin(theta)
    ay_w = ax * np.sin(theta) + ay * np.cos(theta)

    # Integrate acceleration → velocity
    vx = np.cumsum(ax_w * dt)
    vy = np.cumsum(ay_w * dt)

    # Integrate velocity → position increments
    dx = vx * dt
    dy = vy * dt

    dtheta = np.degrees(dtheta)  # Convert to degrees for consistency with GPS bearing

    return dx, dy, dtheta

def calculate_odo_longitude_latitude(start_lat, start_lon, translation_m, rotation_rad):
    """
    Calculate new GPS coordinates given a starting point, translation, and rotation.
    
    Parameters:
    - start_lat: Starting latitude in degrees
    - start_lon: Starting longitude in degrees
    - translation_m: Translation distance in meters [x, y]
    - rotation_rad: Rotation (bearing) in radians (0 rad=North, clockwise)
    
    Returns:
    - new_lat: New latitude in degrees
    - new_lon: New longitude in degrees
    """
    theta_rad = rotation_rad

    # Calculate the change in position in meters
    dx = translation_m[0] * np.cos(theta_rad) - translation_m[1] * np.sin(theta_rad)
    dy = translation_m[0] * np.sin(theta_rad) + translation_m[1] * np.cos(theta_rad)
    
    # Approximate conversion from meters to degrees
    lat_offset = dy / 111320.0  # 1 degree latitude ≈ 111,320 meters
    lon_offset = dx / (111320.0 * np.cos(np.radians(start_lat)))  # Adjust for latitude
    
    new_lat = start_lat + lat_offset
    new_lon = start_lon + lon_offset
    
    return new_lat, new_lon

def rigid_body_transform(
    lat,
    lon,
    heading_deg,
    offset_distance_m=9.0,
    offset_angle_deg=0.0,
):
    """
    Apply a 2D rigid-body lever-arm correction to geodetic coordinates.

    Parameters:
    - lat: Latitude in degrees (sensor source origin)
    - lon: Longitude in degrees (sensor source origin)
    - heading_deg: Compass heading in degrees (0=North, clockwise positive)
    - offset_distance_m: Lever-arm magnitude in meters (default 9 m)
    - offset_angle_deg: Lever-arm angle in body frame in degrees.
        0 deg means "forward" from the source sensor toward the target sensor.

    Returns:
    - corrected_lat: Latitude in degrees at target sensor origin
    - corrected_lon: Longitude in degrees at target sensor origin
    """
    lat = float(lat)
    lon = float(lon)
    heading_rad = np.radians(float(heading_deg))
    alpha_rad = np.radians(float(offset_angle_deg))

    # Body-frame offset from source->target sensor.
    # x_body: forward, y_body: starboard.
    x_body = float(offset_distance_m) * np.cos(alpha_rad)
    y_body = float(offset_distance_m) * np.sin(alpha_rad)

    # Rotate body-frame offset into local ENU world frame.
    east_m = x_body * np.sin(heading_rad) + y_body * np.cos(heading_rad)
    north_m = x_body * np.cos(heading_rad) - y_body * np.sin(heading_rad)

    lat_offset = north_m / 111320.0
    lon_offset = east_m / (111320.0 * np.cos(np.radians(lat)) + 1e-12)

    corrected_lat = lat + lat_offset
    corrected_lon = lon + lon_offset
    return corrected_lat, corrected_lon

def map_gps(
    gps_data,
    odo_trans,
    timestamps,
    odo_rotations_rad=None,
    odo_rotations_deg=None,
    odo_rotations_reference="relative",
    odo_longitudes=None,
    odo_latitudes=None,
    apply_rigid_body_correction=True,
    sensor_offset_m=9.0,
    sensor_offset_angle_deg=0.0,
    output_file="odometry\\gps_map.html",
):
    """
    Visualize GPS points and trajectory on an OpenStreetMap basemap.
    
    Parameters:
    - gps_data: Full DataFrame with GPS data ('latitude', 'longitude', 'time-string')
    - odo_trans: Odometry translations. Supports either:
        * incremental steps (N-1, 2) [dx, dy], or
        * absolute local positions (N, 2)
    - timestamps: Radar timestamps
    - odo_rotations_rad: Optional odometry headings in radians (absolute)
    - odo_rotations_deg: Optional odometry headings in degrees (absolute)
        - odo_rotations_reference: "relative" (default) or "absolute".
            Use "relative" when odometry yaw starts at 0 in frame-1.
    - odo_longitudes: Optional odometry longitudes (direct trajectory input)
    - odo_latitudes: Optional odometry latitudes (direct trajectory input)
        - apply_rigid_body_correction: If True, convert odometry points from radar origin
            to GPS antenna origin for a fair overlay with raw GPS measurements.
        - sensor_offset_m: Lever-arm distance from GPS->radar in meters (default 9 m)
        - sensor_offset_angle_deg: GPS->radar angle in body frame (0=forward)
    - output_file: HTML file path for the generated interactive map
    """
    if timestamps is None or len(timestamps) == 0:
        print("No timestamps provided for map generation.")
        return None

    # Extract GPS data for the relevant timeframe
    gps_subset = extract_timeframe(gps_data, timestamps)
    
    if gps_subset.empty:
        print("No GPS data to visualize on the map.")
        return None

    gps_plot = gps_subset.copy().reset_index(drop=True)
    coordinates = gps_plot[['latitude', 'longitude']].values.tolist()
    bearing = gps_plot['bearing'].values.tolist()

    # Build odometry trajectory by converting translations to GPS coordinates
    # and accumulating heading from odometry rotations
    # Interpolate the initial position to fit the first radar timestamp
    initial_pos = interpolate_gps_motion(gps_subset, [timestamps[0]])
    if initial_pos.empty:
        print("No GPS data available to determine initial position for odometry trajectory.")
        return None
    else:
        initial_lat = initial_pos.iloc[0]['latitude']
        initial_lon = initial_pos.iloc[0]['longitude']
        coordinates.insert(0, [initial_lat, initial_lon])

    initial_heading_deg = bearing[0] if len(bearing) > 0 and pd.notna(bearing[0]) else 0.0
    initial_heading_rad = np.radians(float(initial_heading_deg))

    if odo_rotations_deg is not None:
        odo_headings_rad = np.radians(np.asarray(odo_rotations_deg, dtype=float).reshape(-1))
    elif odo_rotations_rad is not None:
        odo_headings_rad = np.asarray(odo_rotations_rad, dtype=float).reshape(-1)
    else:
        odo_headings_rad = np.asarray([], dtype=float)

    if len(odo_headings_rad) > 0 and str(odo_rotations_reference).lower() == "relative":
        odo_headings_rad = odo_headings_rad + float(initial_heading_rad)

    coordinates_odo = []
    headings_odo = []

    # Path 1: direct odometry GPS trajectory
    if odo_latitudes is not None and odo_longitudes is not None:
        lat_arr = np.asarray(odo_latitudes, dtype=float).reshape(-1)
        lon_arr = np.asarray(odo_longitudes, dtype=float).reshape(-1)
        n_pts = min(len(lat_arr), len(lon_arr))

        if n_pts == 0:
            print("No valid odometry latitude/longitude points provided.")
            return None

        for i in range(n_pts):
            coordinates_odo.append([float(lat_arr[i]), float(lon_arr[i])])

        if len(odo_headings_rad) >= n_pts:
            headings_odo = [float(h) for h in odo_headings_rad[:n_pts]]
        elif len(coordinates_odo) >= 2:
            headings_odo = [initial_heading_rad]
            for i in range(1, len(coordinates_odo)):
                prev = {
                    "latitude": coordinates_odo[i - 1][0],
                    "longitude": coordinates_odo[i - 1][1],
                }
                curr = {
                    "latitude": coordinates_odo[i][0],
                    "longitude": coordinates_odo[i][1],
                }
                headings_odo.append(np.radians(calculate_gps_bearing(prev, curr)))
        else:
            headings_odo = [initial_heading_rad]

    # Path 2: reconstruct odometry GPS trajectory from translations
    else:
        trans_arr = np.asarray(odo_trans, dtype=float)
        if trans_arr.ndim == 1:
            trans_arr = trans_arr.reshape(-1, 2)
        if trans_arr.ndim != 2 or trans_arr.shape[1] < 2:
            raise ValueError("odo_trans must be array-like with shape (N,2) or (N-1,2).")

        trans_arr = trans_arr[:, :2]

        # If trajectory is absolute positions (len == timestamps), convert to step increments.
        if len(trans_arr) == len(timestamps):
            odo_steps = np.diff(trans_arr, axis=0)
        else:
            odo_steps = trans_arr

        n_steps = len(odo_steps)
        if n_steps == 0:
            coordinates_odo = [[coordinates[0][0], coordinates[0][1]]]
            headings_odo = [initial_heading_rad]
        else:
            if len(odo_headings_rad) == n_steps + 1:
                step_headings = odo_headings_rad[1:]
            elif len(odo_headings_rad) == n_steps:
                step_headings = odo_headings_rad
            elif len(odo_headings_rad) > n_steps:
                step_headings = odo_headings_rad[:n_steps]
            elif len(odo_headings_rad) > 0:
                pad = np.full((n_steps - len(odo_headings_rad),), float(odo_headings_rad[-1]), dtype=float)
                step_headings = np.concatenate([odo_headings_rad, pad])
            else:
                step_headings = np.full((n_steps,), initial_heading_rad, dtype=float)

            current_lat, current_lon = coordinates[0]

            coordinates_odo = [[current_lat, current_lon]]
            headings_odo = [float(step_headings[0]) if n_steps > 0 else initial_heading_rad]

            for i, trans in enumerate(odo_steps):
                dx_local, dy_local = float(trans[0]), float(trans[1])
                current_heading_rad = float(step_headings[i])

                current_lat, current_lon = calculate_odo_longitude_latitude(
                    current_lat,
                    current_lon,
                    [dx_local, dy_local],
                    current_heading_rad,
                )
                coordinates_odo.append([current_lat, current_lon])
                headings_odo.append(current_heading_rad)

    # Apply rigid-body correction to convert from radar origin to GPS origin.
    # User-provided lever-arm is GPS->radar, so invert by 180 deg.
    if apply_rigid_body_correction:
        corrected_coordinates_odo = []
        correction_angle_deg = (float(sensor_offset_angle_deg) + 180.0) % 360.0
        for (lat, lon), heading in zip(coordinates_odo, headings_odo):
            corrected_lat, corrected_lon = rigid_body_transform(
                lat,
                lon,
                np.degrees(heading),
                offset_distance_m=sensor_offset_m,
                offset_angle_deg=correction_angle_deg,
            )
            corrected_coordinates_odo.append([corrected_lat, corrected_lon])
        coordinates_odo = corrected_coordinates_odo

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
            tooltip=f"Odometry Point {i} | Heading: {np.degrees(headings_odo[i]):.2f}°"
        ).add_to(fmap)

    if len(coordinates) > 0:
        folium.CircleMarker(
            location=coordinates[0],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=1.0,
            tooltip="GPS Start"
        ).add_to(fmap)
    if len(coordinates) > 1:
        folium.CircleMarker(
            location=coordinates[-1],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=1.0,
            tooltip="GPS End"
        ).add_to(fmap)
    if len(coordinates_odo) > 1:
        folium.CircleMarker(
            location=coordinates_odo[-1],
            radius=5,
            color="green",
            fill=True,
            fill_opacity=1.0,
            tooltip="Odometry End"
        ).add_to(fmap)

        final_heading = float(headings_odo[-1])
        heading_css_deg = np.degrees(final_heading) - 90.0
        arrow_html = (
            f'<div style="font-size: 20px; color: green; font-weight: bold; '
            f'transform: rotate({heading_css_deg:.2f}deg); transform-origin: center;">➤</div>'
        )
        folium.Marker(
            location=coordinates_odo[-1],
            icon=folium.DivIcon(html=arrow_html),
            tooltip=f"Odometry Heading: {np.degrees(final_heading):.2f}°"
        ).add_to(fmap)

    fmap.save(output_file)
    print(f"Saved OpenStreetMap visualization to {output_file}")
    return fmap

if __name__ == "__main__":
    ransac = True

    image_file, image = load_radar_images(num_images=5)
    img1 = image[2]
    img2 = image[3]
    
    timestamp1 = extract_timestamp(image_file[0])
    timestamp2 = extract_timestamp(image_file[1])
    print(f"Timestamp for Image 1: {timestamp1}")
    print(f"Timestamp for Image 2: {timestamp2}")
    
    cartesian_image1, _, _ = polar_to_cartesian_image(img1)
    cartesian_image2, _, _ = polar_to_cartesian_image(img2)
    
    S1, H1 = compute_H_S(img1)
    S2, H2 = compute_H_S(img2)
    
    keypoints1 = Cen2019_keypoints(H1, S1, l_max=200)
    keypoints2 = Cen2019_keypoints(H2, S2, l_max=200)

    descriptors1 = compute_descriptors(img1, keypoints1)
    descriptors2 = compute_descriptors(img2, keypoints2)

    U = unaryMatchesFromDescriptors(descriptors1, descriptors2)
    print(f"Found {len(U)} unary matches between the two images")
    Compatibility = compute_pairwiseCompatibilityScore(U, keypoints1, keypoints2)
    M = select_matches(U, Compatibility)

    if ransac:
        theta_deg, t, points1, points2, inliers = motion_estimation_ransac(M, keypoints1, keypoints2)
        print(f"Rotation:\n{theta_deg}°\nEstimated translation:\n{t} m -> {np.linalg.norm(t):.3f} m")
        
        # Reconstruct best_R for visualization
        best_R = np.array([[np.cos(np.radians(theta_deg)), -np.sin(np.radians(theta_deg))],
                        [np.sin(np.radians(theta_deg)), np.cos(np.radians(theta_deg))]])
        
        # Visualize RANSAC results
        visualize_ransac_motion(points1, points2, inliers, best_R, t, theta_deg)
    else:
        R, t = motion_estimation_SVD(M, keypoints1, keypoints2)
        theta_rad = np.arctan2(R[1, 0], R[0, 0])
        theta_deg = np.degrees(theta_rad) % 360
        print(f"Rotation:\n{theta_deg}°\nEstimated translation:\n{t} m -> {np.linalg.norm(t):.3f} m")
    
    # Visualize transformed image comparison
    visualize_transformed_overlay(img1, img2, theta_deg, t)

    gps_data = load_gps_data()
    #gps_subset = extract_timeframe(gps_data, [timestamp1, timestamp2])
    gps_subset = interpolate_gps_motion(gps_data, [timestamp1, timestamp2])

    gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
    print(f"Ground truth translation from GPS data: {gt_translation} m")
    print(f"Ground truth bearing from GPS data: {gt_bearing} degrees")
    #map_gps(gps_data, [t], [timestamp1, timestamp2], odo_rotations_rad=[np.radians(theta_deg)])
