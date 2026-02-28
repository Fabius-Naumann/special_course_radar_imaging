from data_loading import load_radar_images, load_gps_data, polar_to_cartesian_points, extract_timestamp
from keypoint_extraction import compute_H_S, extract_keypoints
from data_association import compute_descriptors, unaryMatchesFromDescriptors, compute_pairwiseCompatibilityScore, select_matches
from evaluation import calculate_gt_motion, motion_estimation, extract_timeframe

l_max = 200 #maximum number of keypoints

if __name__ == "__main__":
    # Step 1: Load radar images and GPS data
    image_files, images = load_radar_images(num_images=10)
    gps_data = load_gps_data()

    # Step 2: Detect Keypoints and Compute according Descriptors
    print("Image 0:\n")
    S1, H1 = compute_H_S(images[0])
    kp1 = extract_keypoints(H1, S1, l_max=l_max) 
    des1 = compute_descriptors(images[0], kp1)

    timestamps = []
    rot = []
    trans = []

    timestamps.append(extract_timestamp(image_files[0]))

    for point in range(len(image_files)-1):
        print(f"\nImage {point+1}:\n")
        S2, H2 = compute_H_S(images[point+1])
        kp2 = extract_keypoints(H2, S2, l_max=l_max)
        des2 = compute_descriptors(images[point+1], kp2)

        # Step 3: Initial match between keypoints
        U = unaryMatchesFromDescriptors(des1, des2)
        print(f"Found {len(U)} unary matches between the two images")
        
        # Step 4: Optimize matches with Compatibility Score
        C = compute_pairwiseCompatibilityScore(U, kp1, kp2)
        M = select_matches(U, C)

        # Step 5: Estimate the motion based on the matches
        r, t = motion_estimation(M, kp1, kp2)
        print(f"Estimated rotation:\n{r}°\nEstimated translation:\n{t} m")
        rot.append(r)
        trans.append(t)

        # Step 6: Compare to Groundtruth
        timestamps.append(extract_timestamp(image_files[point+1]))
        gps_subset = extract_timeframe(gps_data, [timestamps[point], timestamps[point+1]])
        gt_translation, gt_bearing = calculate_gt_motion(gps_subset)
        print(f"Ground truth translation from GPS data: {gt_translation} m")
        print(f"Ground truth bearing from GPS data: {gt_bearing} degrees")

        # Step 7: Switch variables in preparation for next image
        kp1 = kp2
        des1 = des2 