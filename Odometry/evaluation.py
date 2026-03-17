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

from data_loading import extract_timestamp, load_gps_data, load_radar_images, polar_to_cartesian_image
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
    cart_img1 = polar_to_cartesian_image(img1)
    cart_img2 = polar_to_cartesian_image(img2)
    
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
    fps=8,
    frames_per_image=4,
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
    if len(rotations_deg) != n_frames - 1 or len(translations) != n_frames - 1:
        raise ValueError(
            "'rotations_deg' and 'translations' must both have length len(images)-1."
        )
    if int(frames_per_image) < 1:
        raise ValueError("'frames_per_image' must be >= 1.")

    frames_per_image = int(frames_per_image)

    cart_images = [polar_to_cartesian_image(img).astype(np.float32) for img in images]
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
            theta = np.radians(-float(rotations_deg[i - 1]))
            c, s = np.cos(theta), np.sin(theta)
            tx_m, ty_m = np.asarray(translations[i - 1], dtype=float)[:2]
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

def interpolate_gps_motion(gps_subset, timestamps):
    """
    Interpolate GPS motion to estimate position at specific timestamps.
    
    Parameters:
    - gps_subset: DataFrame containing GPS data for the relevant timeframe
    - timestamps: List of timestamps to interpolate GPS positions for
    
    Returns:
    - interpolated_positions: DataFrame with columns ['time-string', 'latitude', 'longitude']
    """
    interpolated_positions = []
    gps_sorted = gps_subset.sort_values('time-string').reset_index(drop=True)
    
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
                'longitude': nearest['longitude']
            })
            continue

        if after_candidates.empty:
            nearest = before_candidates.iloc[-1]
            interpolated_positions.append({
                'time-string': radar_time_point,
                'latitude': nearest['latitude'],
                'longitude': nearest['longitude']
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
                'longitude': before['longitude']
            })
        else:
            # Linear interpolation based on time
            total_time_diff = (after['time-string'] - before['time-string']).total_seconds()
            if total_time_diff == 0:
                interpolated_positions.append({
                    'time-string': radar_time_point,
                    'latitude': before['latitude'],
                    'longitude': before['longitude']
                })
                continue
            
            time_diff_to_before = (radar_time_point - before['time-string']).total_seconds()
            ratio = time_diff_to_before / total_time_diff
            
            lat_interp = before['latitude'] + ratio * (after['latitude'] - before['latitude'])
            lon_interp = before['longitude'] + ratio * (after['longitude'] - before['longitude'])
            
            interpolated_positions.append({
                'time-string': radar_time_point,
                'latitude': lat_interp,
                'longitude': lon_interp
            })
    
    return pd.DataFrame(interpolated_positions)

def map_gps(gps_data, odo_trans, timestamps, odo_rotations_deg=None, output_file="odometry\gps_map.html"):
    """
    Visualize GPS points and trajectory on an OpenStreetMap basemap.
    
    Parameters:
    - gps_data: Full DataFrame containing GPS data with 'latitude', 'longitude', and 'time-string' fields
    - odo_trans: List of odometry translations in local frame [dx, dy]
    - timestamps: List of timestamps to extract GPS data for
    - odo_rotations_deg: Optional list of incremental yaw rotations (degrees), one per odometry step
    - output_file: HTML file path for the generated interactive map
    """
    # Extract GPS data for the relevant timeframe
    gps_subset = extract_timeframe(gps_data, timestamps)
    
    if gps_subset.empty:
        print("No GPS data to visualize on the map.")
        return

    gps_plot = gps_subset.copy().reset_index(drop=True)
    coordinates = gps_plot[['latitude', 'longitude']].values.tolist()
    bearing = gps_plot['bearing'].values.tolist()

    # Build odometry trajectory by converting translations to GPS coordinates
    # and accumulating heading from odometry rotations
    # Start from first GPS coordinate
    coordinates_odo = [coordinates[0]]
    current_lat, current_lon = coordinates[0]
    current_heading_deg = bearing[0] if len(bearing) > 0 else 0.0
    headings_odo = [current_heading_deg]
    
    # Convert odometry translations (in meters) to GPS coordinate offsets
    for i, trans in enumerate(odo_trans):
        # Handle both numpy arrays and regular lists
        if hasattr(trans, '__iter__') and len(trans) >= 2:
            dx_local, dy_local = float(trans[0]), float(trans[1])

            # Rotate local translation into global map frame.
            # current_heading_deg is compass heading (0°=North, clockwise),
            # so convert to math angle (0°=East, counterclockwise).
            heading_math_rad = np.radians(90.0 - current_heading_deg)
            dx_global = np.cos(heading_math_rad) * dx_local - np.sin(heading_math_rad) * dy_local
            dy_global = np.sin(heading_math_rad) * dx_local + np.cos(heading_math_rad) * dy_local
            
            # Approximate conversion: 1 degree latitude ≈ 111,320 meters
            # 1 degree longitude varies with latitude: ≈ 111,320 * cos(latitude)
            lat_offset = dy_global / 111320.0
            lon_offset = dx_global / (111320.0 * np.cos(np.radians(current_lat)))
            
            current_lat += lat_offset
            current_lon += lon_offset
            coordinates_odo.append([current_lat, current_lon])

            if odo_rotations_deg is not None and i < len(odo_rotations_deg):
                # Odometry yaw is treated as math-positive (CCW). Convert to compass update.
                delta_heading_deg = ((float(odo_rotations_deg[i]) + 180.0) % 360.0) - 180.0
                current_heading_deg = (current_heading_deg - delta_heading_deg) % 360.0
            headings_odo.append(current_heading_deg)
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
            tooltip=f"Odometry Point {i} | Heading: {headings_odo[i]:.2f}°"
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
        heading_css_deg = final_heading - 90.0
        arrow_html = (
            f'<div style="font-size: 20px; color: green; font-weight: bold; '
            f'transform: rotate({heading_css_deg:.2f}deg); transform-origin: center;">➤</div>'
        )
        folium.Marker(
            location=coordinates_odo[-1],
            icon=folium.DivIcon(html=arrow_html),
            tooltip=f"Odometry Heading: {final_heading:.2f}°"
        ).add_to(fmap)

    fmap.save(output_file)
    print(f"Saved OpenStreetMap visualization to {output_file}")

if __name__ == "__main__":
    ransac = True

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
    #map_gps(gps_data, [t], [timestamp1, timestamp2], odo_rotations_deg=[theta_deg])
