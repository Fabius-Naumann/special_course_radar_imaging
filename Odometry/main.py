from data_loading import load_radar_images, load_gps_data, extract_timestamp
from keypoint_extraction import compute_H_S, Cen2019_keypoints, k_strongest_keypoints
from descriptors import compute_descriptors, estimate_oriented_surface_points, orb_descriptor, radial_statistics_descriptor
from data_association import (
    unaryMatchesFromDescriptors,
    compute_pairwiseCompatibilityScore, select_matches,
    registration_from_oriented_points
)
from evaluation import calculate_gt_motion, map_gps, interpolate_gps_motion, create_stabilized_overlay_video
from motion_estimation import (
    motion_estimation_ransac,
    motion_estimation_SVD,
    motion_estimation_bundle_adjustment,
)
import numpy as np
import os
import pandas as pd

# Configuration
USE_CFEAR_METHOD = True  # Set to True for CFEAR-3 oriented points, False for keypoint matching
l_max = 300  # maximum number of keypoints
r_param = 5.0  # radius for surface point estimation (meters)
f_param = 1.0  # downsampling factor for oriented points
n_images = 15  # number of radar images to process
DESCRIPTOR_TYPE = "cen2019"  # Options: "cen2019", "orb", "radial"
KEYPOINT_TYPE = "cen2019"  # Options: "cen2019", "k_strongest", "blob", "orb"


def build_descriptors(img, keypoints, descriptor_type):
    """
    Build descriptors for keypoints and return aligned (keypoints, descriptors).
    """
    keypoints = np.asarray(keypoints)
    if keypoints.size == 0:
        return keypoints.reshape(0, 2), np.empty((0, 0), dtype=float)

    mode = str(descriptor_type).strip().lower()

    if mode in ("cen2019", "default", "hist"):
        descriptors = np.asarray(compute_descriptors(img, keypoints), dtype=float)
        return keypoints, descriptors

    if mode == "orb":
        descriptors, valid_indices = orb_descriptor(img, keypoints)
        if valid_indices.size == 0:
            return np.empty((0, 2), dtype=float), np.empty((0, 32), dtype=float)
        return keypoints[valid_indices], np.asarray(descriptors, dtype=float)

    if mode in ("radial", "radial_statistics", "rsd"):
        descriptors = radial_statistics_descriptor(keypoints)
        descriptors = np.asarray(descriptors, dtype=float).reshape(len(keypoints), -1)
        return keypoints, descriptors

    raise ValueError(
        f"Unsupported DESCRIPTOR_TYPE='{descriptor_type}'. "
        "Use one of: 'cen2019', 'orb', 'radial'."
    )

if __name__ == "__main__":
    # Step 1: Load radar images and GPS data
    image_files, images = load_radar_images(num_images=n_images)
    gps_data = load_gps_data()

    timestamps = []
    rot = []
    trans = []
    
    print(f"Using {'CFEAR-3 Oriented Points' if USE_CFEAR_METHOD else 'Keypoint Matching'} method")
    if not USE_CFEAR_METHOD:
        print(f"Using keypoint extraction: {KEYPOINT_TYPE}")
        print(f"Descriptor type: {DESCRIPTOR_TYPE}")
    print(f"Processing {len(image_files)} images...\n")

    if USE_CFEAR_METHOD:
        # ============ CFEAR-3 METHOD: Oriented Surface Points ============
        print("Image 0: Estimating oriented surface points")
        kp1 = k_strongest_keypoints(images[0], z_min=0.55, k=20)  # Filter keypoints to top k strongest
        oriented_points1, _ = estimate_oriented_surface_points(
            images[0], kp1, r=r_param, f=f_param
        )
        print(f"  Extracted {len(kp1)} keypoints, created {len(oriented_points1)} oriented surface points")
        
        timestamps.append(extract_timestamp(image_files[0]))

        for point in range(len(images) - 1):
            print(f"\nImage {point+1}: Estimating oriented surface points")
            kp2 = k_strongest_keypoints(images[point+1], z_min=0.55, k=20)
            oriented_points2, _ = estimate_oriented_surface_points(
                images[point+1], kp2, r=r_param, f=f_param
            )
            print(f"  Extracted {len(kp2)} keypoints, created {len(oriented_points2)} oriented surface points")

            # Step 3: Register oriented points to estimate motion
            if len(oriented_points1) >= 3 and len(oriented_points2) >= 3:
                R, t, correspondences, cov = registration_from_oriented_points(
                    (oriented_points1, oriented_points2),
                    return_covariance=True
                )
                
                # Registration aligns points from scan1 to scan2:
                # p2 = R * p1 + t. For ego-motion integration, use inverse transform.
                R_ego = R.T
                t_ego = -R_ego @ np.asarray(t, dtype=float)

                # Signed incremental ego yaw (math convention, CCW positive)
                theta_rad = np.arctan2(R_ego[1, 0], R_ego[0, 0])
                theta_deg = np.degrees(theta_rad)
                
                print(f"  Estimated rotation: {theta_deg:.2f}°")
                print(f"  Estimated translation: [{t_ego[0]:.3f}, {t_ego[1]:.3f}] m")
                print(f"  Pose uncertainty (trace): {np.trace(cov):.6f}")
                print(f"  Number of correspondences: {len(correspondences)}")
                
                rot.append(theta_deg)
                trans.append(t_ego)
            else:
                print(f"  Warning: Not enough oriented points for registration")
                rot.append(0.0)
                trans.append(np.array([0.0, 0.0]))

            # Step 4: Compare to Groundtruth
            timestamps.append(extract_timestamp(image_files[point+1]))
            gps_subset = interpolate_gps_motion(gps_data, [timestamps[point], timestamps[point+1]])
            gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
            print(f"  Ground truth translation: {gt_translation} m")
            print(f"  Ground truth bearing: {gt_bearing}°")

            # Step 5: Switch variables for next iteration
            oriented_points1 = oriented_points2
            kp1 = kp2

    else:
        # ============ ORIGINAL METHOD: Keypoint Matching ============
        print("Image 0: Detecting keypoints")
        S1, H1 = compute_H_S(images[0])
        kp1 = Cen2019_keypoints(H1, S1, l_max=l_max) 
        kp1, des1 = build_descriptors(images[0], kp1, DESCRIPTOR_TYPE)
        print(f"  Extracted {len(kp1)} keypoints with descriptors")

        timestamps.append(extract_timestamp(image_files[0]))

        for point in range(len(images)-1):
            print(f"\nImage {point+1}: Detecting keypoints")
            S2, H2 = compute_H_S(images[point+1])
            kp2 = Cen2019_keypoints(H2, S2, l_max=l_max)
            kp2, des2 = build_descriptors(images[point+1], kp2, DESCRIPTOR_TYPE)

            print(f"  Extracted {len(kp2)} keypoints with descriptors")

            if len(kp1) == 0 or len(kp2) == 0 or len(des1) == 0 or len(des2) == 0:
                print("  Warning: Missing descriptors/keypoints, skipping motion estimation for this step")
                rot.append(0.0)
                trans.append(np.array([0.0, 0.0]))

                timestamps.append(extract_timestamp(image_files[point+1]))
                gps_subset = interpolate_gps_motion(gps_data, [timestamps[point], timestamps[point+1]])
                gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
                print(f"  Ground truth translation: {gt_translation} m")
                print(f"  Ground truth bearing: {gt_bearing}°")

                kp1 = kp2
                des1 = des2
                continue

            # Step 3: Initial match between keypoints
            U = unaryMatchesFromDescriptors(des1, des2)
            print(f"  Found {len(U)} unary matches")

            if len(U) == 0:
                print("  Warning: No unary matches, skipping motion estimation for this step")
                rot.append(0.0)
                trans.append(np.array([0.0, 0.0]))

                timestamps.append(extract_timestamp(image_files[point+1]))
                gps_subset = interpolate_gps_motion(gps_data, [timestamps[point], timestamps[point+1]])
                gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
                print(f"  Ground truth translation: {gt_translation} m")
                print(f"  Ground truth bearing: {gt_bearing}°")

                kp1 = kp2
                des1 = des2
                continue
            
            # Step 4: Optimize matches with Compatibility Score
            C = compute_pairwiseCompatibilityScore(U, kp1, kp2)
            M = select_matches(U, C)

            if len(M) == 0:
                print("  Warning: No compatible matches selected, skipping motion estimation for this step")
                rot.append(0.0)
                trans.append(np.array([0.0, 0.0]))

                timestamps.append(extract_timestamp(image_files[point+1]))
                gps_subset = interpolate_gps_motion(gps_data, [timestamps[point], timestamps[point+1]])
                gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
                print(f"  Ground truth translation: {gt_translation} m")
                print(f"  Ground truth bearing: {gt_bearing}°")

                kp1 = kp2
                des1 = des2
                continue

            # Step 5: Estimate the motion based on the matches
            r, t, _, _, _ = motion_estimation_ransac(M, kp1, kp2)

            # Registration aligns points from scan1 to scan2:
            # p2 = R * p1 + t. Convert to ego-motion increment via inverse transform.
            theta_points_deg = ((float(r) + 180.0) % 360.0) - 180.0
            theta_points_rad = np.radians(theta_points_deg)
            R_points = np.array([
                [np.cos(theta_points_rad), -np.sin(theta_points_rad)],
                [np.sin(theta_points_rad),  np.cos(theta_points_rad)]
            ])
            R_ego = R_points.T
            t_ego = -R_ego @ np.asarray(t, dtype=float)
            theta_ego_deg = np.degrees(np.arctan2(R_ego[1, 0], R_ego[0, 0]))

            print(f"  Estimated rotation: {theta_ego_deg:.2f}°")
            print(f"  Estimated translation: {t_ego} m")
            rot.append(theta_ego_deg)
            trans.append(t_ego)

            # Step 6: Compare to Groundtruth
            timestamps.append(extract_timestamp(image_files[point+1]))
            gps_subset = interpolate_gps_motion(gps_data, [timestamps[point], timestamps[point+1]])
            gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
            print(f"  Ground truth translation: {gt_translation} m")
            print(f"  Ground truth bearing: {gt_bearing}°")

            # Step 7: Switch variables in preparation for next image
            kp1 = kp2
            des1 = des2 

    print("\n" + "="*60)
    # Step 7: Save results and generate GPS comparison map
    output_dir = "odometry"
    os.makedirs(output_dir, exist_ok=True)

    rot_array = np.asarray(rot, dtype=float)
    trans_array = np.asarray(trans, dtype=float)
    if trans_array.ndim == 1:
        trans_array = trans_array.reshape(-1, 2)

    np.savez(
        os.path.join(output_dir, "odometry_results.npz"),
        timestamps=np.asarray(timestamps, dtype=object),
        rotations_deg=rot_array,
        translations_xy=trans_array,
    )

    step_count = min(len(rot_array), trans_array.shape[0], max(0, len(timestamps) - 1))
    results_df = pd.DataFrame(
        {
            "timestamp_start": timestamps[:step_count],
            "timestamp_end": timestamps[1:step_count + 1],
            "rotation_deg": rot_array[:step_count],
            "translation_x_m": trans_array[:step_count, 0],
            "translation_y_m": trans_array[:step_count, 1],
        }
    )
    results_df.to_csv(os.path.join(output_dir, "odometry_results.csv"), index=False)
    print(f"Saved odometry estimates to {os.path.join(output_dir, 'odometry_results.npz')} and {os.path.join(output_dir, 'odometry_results.csv')}")

    print("Generating stabilized transformed overlay video...")
    create_stabilized_overlay_video(
        images=images,
        rotations_deg=rot,
        translations=trans,
        output_file=os.path.join(output_dir, "transformed_overlay.mp4"),
        fps=8,
    )

    print("Generating GPS comparison map...")
    map_gps(gps_data, trans, timestamps, odo_rotations_deg=rot)