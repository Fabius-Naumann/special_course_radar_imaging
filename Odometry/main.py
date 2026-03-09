from data_loading import load_radar_images, load_gps_data, polar_to_cartesian_points, extract_timestamp
from keypoint_extraction import compute_H_S, extract_keypoints, k_strongest_keypoints
from data_association import (
    compute_descriptors, unaryMatchesFromDescriptors, 
    compute_pairwiseCompatibilityScore, select_matches,
    estimate_oriented_surface_points, registration_from_oriented_points
)
from evaluation import calculate_gt_motion, motion_estimation, extract_timeframe, map_gps
import numpy as np

# Configuration
USE_CFEAR_METHOD = True  # Set to True for CFEAR-3 oriented points, False for keypoint matching
l_max = 200  # maximum number of keypoints
r_param = 5.0  # radius for surface point estimation (meters)
f_param = 1.0  # downsampling factor for oriented points
n_images = 30  # number of radar images to process

if __name__ == "__main__":
    # Step 1: Load radar images and GPS data
    image_files, images = load_radar_images(num_images=n_images)
    gps_data = load_gps_data()

    timestamps = []
    rot = []
    trans = []
    
    print(f"Using {'CFEAR-3 Oriented Points' if USE_CFEAR_METHOD else 'Keypoint Matching'} method")
    print(f"Processing {len(image_files)} images...\n")

    if USE_CFEAR_METHOD:
        # ============ CFEAR-3 METHOD: Oriented Surface Points ============
        print("Image 0: Estimating oriented surface points")
        kp1 = k_strongest_keypoints(images[0], z_min=0.6, k=20)  # Filter keypoints to top k strongest
        oriented_points1 = estimate_oriented_surface_points(
            images[0], kp1, r=r_param, f=f_param
        )
        print(f"  Extracted {len(kp1)} keypoints, created {len(oriented_points1)} oriented surface points")
        
        timestamps.append(extract_timestamp(image_files[0]))

        for point in range(len(images) - 1):
            print(f"\nImage {point+1}: Estimating oriented surface points")
            kp2 = k_strongest_keypoints(images[point+1], z_min=0.6, k=20)
            oriented_points2 = estimate_oriented_surface_points(
                images[point+1], kp2, r=r_param, f=f_param
            )
            print(f"  Extracted {len(kp2)} keypoints, created {len(oriented_points2)} oriented surface points")

            # Step 3: Register oriented points to estimate motion
            if len(oriented_points1) >= 3 and len(oriented_points2) >= 3:
                R, t, correspondences, cov = registration_from_oriented_points(
                    (oriented_points1, oriented_points2),
                    return_covariance=True
                )
                
                # Extract rotation angle in degrees
                theta_rad = np.arctan2(R[1, 0], R[0, 0])
                theta_deg = np.degrees(theta_rad) % 360
                
                print(f"  Estimated rotation: {theta_deg:.2f}°")
                print(f"  Estimated translation: [{t[0]:.3f}, {t[1]:.3f}] m")
                print(f"  Pose uncertainty (trace): {np.trace(cov):.6f}")
                print(f"  Number of correspondences: {len(correspondences)}")
                
                rot.append(theta_deg)
                trans.append(t)
            else:
                print(f"  Warning: Not enough oriented points for registration")
                rot.append(0.0)
                trans.append(np.array([0.0, 0.0]))

            # Step 4: Compare to Groundtruth
            timestamps.append(extract_timestamp(image_files[point+1]))
            gps_subset = extract_timeframe(gps_data, [timestamps[point], timestamps[point+1]])
            if len(gps_subset) > 1:
                gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
                # Handle list returns from calculate_gt_motion
                if isinstance(gt_translation, list) and len(gt_translation) > 0:
                    gt_trans_total = sum(gt_translation)
                    gt_bear_avg = sum(gt_bearing) / len(gt_bearing) if len(gt_bearing) > 0 else 0.0
                    print(f"  Ground truth translation: {gt_trans_total:.6f} m")
                    print(f"  Ground truth bearing: {gt_bear_avg:.2f}°")
                elif gt_translation is not None:
                    print(f"  Ground truth translation: {gt_translation:.6f} m")
                    print(f"  Ground truth bearing: {gt_bearing:.2f}°")
            else:
                print(f"  Warning: Not enough GPS data for ground truth comparison")

            # Step 5: Switch variables for next iteration
            oriented_points1 = oriented_points2
            kp1 = kp2

    else:
        # ============ ORIGINAL METHOD: Keypoint Matching ============
        print("Image 0: Detecting keypoints")
        S1, H1 = compute_H_S(images[0])
        kp1 = extract_keypoints(H1, S1, l_max=l_max) 
        des1 = compute_descriptors(images[0], kp1)
        print(f"  Extracted {len(kp1)} keypoints")

        timestamps.append(extract_timestamp(image_files[0]))

        for point in range(len(images)-1):
            print(f"\nImage {point+1}: Detecting keypoints")
            S2, H2 = compute_H_S(images[point+1])
            kp2 = extract_keypoints(H2, S2, l_max=l_max)
            des2 = compute_descriptors(images[point+1], kp2)
            print(f"  Extracted {len(kp2)} keypoints")

            # Step 3: Initial match between keypoints
            U = unaryMatchesFromDescriptors(des1, des2)
            print(f"  Found {len(U)} unary matches")
            
            # Step 4: Optimize matches with Compatibility Score
            C = compute_pairwiseCompatibilityScore(U, kp1, kp2)
            M = select_matches(U, C)

            # Step 5: Estimate the motion based on the matches
            r, t, _, _, _ = motion_estimation(M, kp1, kp2)
            print(f"  Estimated rotation: {r}°")
            print(f"  Estimated translation: {t} m")
            rot.append(r)
            trans.append(t)

            # Step 6: Compare to Groundtruth
            timestamps.append(extract_timestamp(image_files[point+1]))
            gps_subset = extract_timeframe(gps_data, [timestamps[point], timestamps[point+1]])
            if len(gps_subset) > 1:
                gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
                # Handle list returns from calculate_gt_motion
                if isinstance(gt_translation, list) and len(gt_translation) > 0:
                    gt_trans_total = sum(gt_translation)
                    gt_bear_avg = sum(gt_bearing) / len(gt_bearing) if len(gt_bearing) > 0 else 0.0
                    print(f"  Ground truth translation: {gt_trans_total:.6f} m")
                    print(f"  Ground truth bearing: {gt_bear_avg:.2f}°")
                elif gt_translation is not None:
                    print(f"  Ground truth translation: {gt_translation:.6f} m")
                    print(f"  Ground truth bearing: {gt_bearing:.2f}°")
            else:
                print(f"  Warning: Not enough GPS data for ground truth comparison")

            # Step 7: Switch variables in preparation for next image
            kp1 = kp2
            des1 = des2 

    print("\n" + "="*60)
    print("Generating GPS comparison map...")
    map_gps(gps_data, trans, timestamps)