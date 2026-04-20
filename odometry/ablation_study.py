import numpy as np
import pandas as pd

def main_odometry(n_images, preprocessing, k, z_percentile, max_distance_percentile, r_param, f_param, cost_function, ICP, window_size, motion_compensation_flag, print_lines=False, plot_map=False, smoothing=None, dataset=1):
    """Load the radar images and GPS data"""
    if dataset == 1:
        dataset_dir = os.path.join(DATA_DIR, "_radar_data_b_scan_image")
    elif dataset == 2:
        dataset_dir = os.path.join(DATA_DIR, "_radar_data_b_scan_image_2")
    else:
        raise ValueError("Invalid dataset selection. Please choose 1 or 2.")
    
    image_file, image = load_radar_images(num_images=n_images, correct_black_lines_flag=True, folder_path=dataset_dir)
    gps_data = load_gps_data()

    """Initialization of results storage"""
    odometry_results = pd.DataFrame(
        columns=[
            "timestamp",
            "rotation_deg",
            "translation_x_m",
            "translation_y_m",
            "longitude",
            "latitude",
            "velocity_x_m_s",
            "velocity_y_m_s",
            "velocity_theta_deg_s",
            "pos_fault_m"
        ]
    )

    timestamps = []
    keyframes = []
    last_kf_longitude = None
    last_kf_latitude = None

    """Processing of the first valid frame"""
    oriented_points1 = []
    R_1_f = np.eye(2)
    t_1_f = np.zeros(2, dtype=float)
    used_images = []
    heading_offset_deg = 0.0
    frames_map = []

    i = 0

    while len(oriented_points1) == 0:
        if print_lines : print("Computing CFEAR features for the first image...")
        oriented_points1 = computing_CFEAR_Features(image[i], preprocessing, k, z_percentile, max_distance_percentile, [0, 0, 0], r_param, f_param, motion_compensation_flag=False, smoothing=smoothing)

        if len(oriented_points1) == 0:
            if print_lines : print("No oriented points extracted from the first image, skipping that image...")
            i += 1
            if i >= n_images:
                raise ValueError("No oriented points could be extracted from any of the images. Please check the data and parameters.")
        else:
            if print_lines : print(f"Successfully extracted {len(oriented_points1)} oriented points from image {i+1}.")

            # Take the GPS data at the corresponding time as initial values for the first frame
            timestamp = extract_timestamp(image_file[i])
            gps_df = interpolate_gps_motion(gps_data, [timestamp])

            # GPS bearing is compass convention (0°=North, clockwise).
            # Convert to math yaw used by the odometry pipeline (0°=+x/East, counterclockwise).
            initial_bearing_compass = float(gps_df["bearing"].values[0])
            if np.isnan(initial_bearing_compass):
                initial_theta_deg = 0.0
                if print_lines : print("Warning: initial GPS bearing is NaN; using 0.0° for initialization.")
            else:
                initial_theta_deg = 90.0 - initial_bearing_compass

            heading_offset_deg = float(initial_theta_deg)

            initial_theta_rad = np.radians(initial_theta_deg)
            R_1_f = np.array([
                [np.cos(initial_theta_rad), -np.sin(initial_theta_rad)],
                [np.sin(initial_theta_rad), np.cos(initial_theta_rad)],
            ], dtype=float)

            # Add that data to the odometry results dataframe
            first_row = pd.DataFrame([
                {
                    "timestamp": timestamp,
                    "rotation_deg": float(initial_theta_deg),
                    "translation_x_m": 0.0,
                    "translation_y_m": 0.0,
                    "longitude": float(gps_df["longitude"].values[0]),
                    "latitude": float(gps_df["latitude"].values[0]),
                    "velocity_x_m_s": 0.0,
                    "velocity_y_m_s": 0.0,
                    "velocity_theta_deg_s": 0.0,
                    "pos_fault_m": 0.0,
                }
            ])
            odometry_results = pd.concat([odometry_results, first_row], ignore_index=True)
            timestamps.append(timestamp)
            used_images.append(image[i])

            # Always add the first successfully processed frame as the initial keyframe
            keyframes.append(oriented_points1)

            last_kf_latitude = float(gps_df["latitude"].values[0])
            last_kf_longitude = float(gps_df["longitude"].values[0])

            i += 1

    for img in range(i, n_images):
        """Compute the features of the current frame"""
        # Extract the velocity from the DataFrame for motion compensation in the current frame
        velocity = (
            float(odometry_results["velocity_x_m_s"].values[-1]),
            float(odometry_results["velocity_y_m_s"].values[-1]),
            float(odometry_results["velocity_theta_deg_s"].values[-1]),
        )

        # Computing CFEAR features for the current image
        if print_lines : print(f"Processing image {img+1}/{n_images}...")
        oriented_points2 = computing_CFEAR_Features(image[img], preprocessing, k, z_percentile, max_distance_percentile, velocity, r_param, f_param, motion_compensation_flag=motion_compensation_flag, smoothing=smoothing)

        if print_lines : print(f"Extracted {len(oriented_points2)} oriented surface points from image {img+1}.")
        if len(oriented_points2) == 0:
            if print_lines : print(f"No oriented points extracted from image {img+1}, skipping registration.")
            continue

        # Initial guess based on last pose and last velocity

        # Compute time difference dt between the current frame and the previous frame
        curr_timestamp = extract_timestamp(image_file[img])
        prev_timestamp = str(odometry_results["timestamp"].values[-1])

        t_prev = datetime.fromisoformat(prev_timestamp)
        t_curr = datetime.fromisoformat(curr_timestamp)

        dt = (t_curr - t_prev).total_seconds()

        initial_theta = np.radians(odometry_results["rotation_deg"].values[-1] - heading_offset_deg + velocity[2]) * dt
        initial_t = np.array(
            [
                float(odometry_results["translation_x_m"].values[-1]),
                float(odometry_results["translation_y_m"].values[-1]),
            ],
            dtype=float,
        )
        initial_t += np.array([velocity[0], velocity[1]]) * dt

        if print_lines : print(f"Initial guess for registration: rotation (degrees): {np.degrees(initial_theta + np.radians(heading_offset_deg)):.2f}, translation (meters): ({initial_t[0]:.2f}, {initial_t[1]:.2f})")

        """Registration and Pose Estimation"""
        # Multi-keyframe mode returns a single absolute transform in the keyframe map coordinates.
        # Because keyframes are already stored in frame-1 coordinates, this is directly current -> frame-1.
        R_f_kf, t_f_kf, correspondences, cov = registration_from_oriented_points(
            oriented_points2, keyframes, cost_function=cost_function, return_covariance=True,
            start_theta=initial_theta, start_t=initial_t, max_iterations=iterations
        )

        if len(correspondences) == 0:
            if print_lines : print(f"No correspondences found for image {img+1}, skipping pose update.")
            continue

        R_f_kf = np.asarray(R_f_kf, dtype=float)
        t_f_kf = np.asarray(t_f_kf, dtype=float)

        theta_rad = np.arctan2(R_f_kf[1, 0], R_f_kf[0, 0])
        theta_deg = float(np.degrees(theta_rad))
        theta_deg = ((theta_deg + heading_offset_deg + 180.0) % 360.0) - 180.0
        theta_rad = np.radians(theta_deg)

        if print_lines : print(f"Initial registration results: rotation (degrees): {theta_deg:.2f}, translation (meters): ({t_f_kf[0]:.2f}, {t_f_kf[1]:.2f}), correspondences: {len(correspondences)}")

        # Previous frame pose in frame-1 coordinates from the last accepted odometry row
        prev_theta_deg = float(odometry_results["rotation_deg"].values[-1])
        prev_theta = np.radians(prev_theta_deg)
        
        c_prev, s_prev = np.cos(prev_theta), np.sin(prev_theta)
        R_1_prev = np.array([[c_prev, -s_prev], [s_prev, c_prev]], dtype=float)
        t_1_prev = np.array([
            float(odometry_results["translation_x_m"].values[-1]),
            float(odometry_results["translation_y_m"].values[-1]),
        ], dtype=float)

        if ICP:
            points1 = oriented_points1[:, :2]
            points2 = oriented_points2[:, :2]
            # Add empty z-axis for Open3D compatibility
            points1 = np.hstack((points1, np.zeros((points1.shape[0], 1))))
            points2 = np.hstack((points2, np.zeros((points2.shape[0], 1))))

            # Convert absolute estimate to current -> previous as ICP initialization (row-vector convention).
            R_prev_1 = R_1_prev.T
            t_prev_1 = -t_1_prev @ R_1_prev
            R_f_prev_init = R_prev_1 @ R_1_f
            t_f_prev_init = t_1_f @ R_prev_1.T + t_prev_1

            transform = ICP_registration(points2, points1, R_init=R_f_prev_init, t_init=t_f_prev_init, distance_threshold=0.2)
            R_f_prev = transform[:2, :2]
            t_f_prev = transform[:2, 3]

            # Compose current -> previous with previous -> frame1 (row-vector convention)
            R_1_f = R_1_prev @ R_f_prev
            t_1_f = t_f_prev @ R_1_prev.T + t_1_prev

            theta_rad = np.arctan2(R_1_f[1, 0], R_1_f[0, 0])
            theta_deg = float(np.degrees(theta_rad))
            theta_deg = ((theta_deg + heading_offset_deg + 180.0) % 360.0) - 180.0
            theta_rad = np.radians(theta_deg)
            if print_lines : print(f"ICP refinement applied. Updated rotation (degrees): {theta_deg:.2f}, translation (meters): ({t_1_f[0]:.2f}, {t_1_f[1]:.2f})")
        else:
            R_1_f = R_f_kf 
            t_1_f = t_f_kf

        # Transform current oriented points into frame-1 coordinate system
        oriented_points2_T = transform_oriented_points(oriented_points2, R_1_f, t_1_f)

        # Estimate longitude and latitude for the current frame from last known position
        last_longitude = float(odometry_results["longitude"].values[-1])
        last_latitude = float(odometry_results["latitude"].values[-1])
        delta_global = t_1_f - np.array([
            float(odometry_results["translation_x_m"].values[-1]),
            float(odometry_results["translation_y_m"].values[-1]),
        ], dtype=float)
        latitude, longitude = calculate_odo_longitude_latitude(
            last_latitude, last_longitude, delta_global, theta_rad
        )

        gps_entry_kf = {"latitude": float(last_kf_latitude), "longitude": float(last_kf_longitude)}
        gps_entry_curr = {"latitude": float(latitude), "longitude": float(longitude)}
        keyframe_distance_m = calculate_gps_distance(gps_entry_kf, gps_entry_curr)

        if keyframe_distance_m > 1.0:
            keyframes.append(oriented_points2_T)
            last_kf_latitude = float(latitude)
            last_kf_longitude = float(longitude)
            if print_lines : print(f"Added image {img+1} as a new keyframe.")
            if len(keyframes) > window_size:
                # Remove the oldest keyframe to maintain the window size
                keyframes.pop(0)
        else:
            if print_lines : print(f"Image {img+1} not added as a keyframe")

        prev_pos = np.array([
            float(odometry_results["translation_x_m"].values[-1]),
            float(odometry_results["translation_y_m"].values[-1]),
        ], dtype=float)
        step_xy = t_1_f - prev_pos
        step_theta = ((theta_deg - prev_theta_deg + 180.0) % 360.0) - 180.0

        if dt > 0.01:
            velocity = (
                float(step_xy[0]) / dt,
                float(step_xy[1]) / dt,
                float(step_theta) / dt,
            )
        else:
            velocity = (0.0, 0.0, 0.0)

        # Compare with GPS ground truth by comparing the longitude and latitude estimated from the odometry with the ones from the GPS at the current timestamp
        gps_df = interpolate_gps_motion(gps_data, [curr_timestamp])
        gt_latitude = gps_df["latitude"].values[0]
        gt_longitude = gps_df["longitude"].values[0]
        position_error = calculate_gps_distance(
            {"latitude": float(gt_latitude), "longitude": float(gt_longitude)},
            {"latitude": float(latitude), "longitude": float(longitude)},
        )
        if print_lines :
            print(f"GPS Ground Truth for current timestamp: latitude: {gt_latitude:.6f}, longitude: {gt_longitude:.6f}")
            print(f"Estimated position: latitude: {latitude:.6f}, longitude: {longitude:.6f}")
            print(f"Position error compared to GPS: {position_error:.2f} m")
            print("-" * 50)

    #
        # Append current frame to odometry results
        current_row = pd.DataFrame([
            {
                "timestamp": curr_timestamp,
                "rotation_deg": theta_deg,
                "translation_x_m": float(t_1_f[0]),
                "translation_y_m": float(t_1_f[1]),
                "longitude": float(longitude),
                "latitude": float(latitude),
                "velocity_x_m_s": float(velocity[0]),
                "velocity_y_m_s": float(velocity[1]),
                "velocity_theta_deg_s": float(velocity[2]),
                "pos_fault_m": float(position_error),
            }
        ])
        odometry_results = pd.concat([odometry_results, current_row], ignore_index=True)
        timestamps.append(curr_timestamp)
        used_images.append(image[img])

        # create images for visualization map from the current oriented points and the estimated pose
        if plot_map:
            if plot_map:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_title(f"Oriented Points Map - Image {img+1}")
                if len(keyframes) > 0:
                    keyframe_points = np.vstack([kf[:, :2] for kf in keyframes if len(kf) > 0])
                    ax.scatter(keyframe_points[:, 0], keyframe_points[:, 1], s=5, c="red", label="Current Keyframes")
                if len(oriented_points2_T) > 0:
                    ax.scatter(oriented_points2_T[:, 0], oriented_points2_T[:, 1], s=5, c="blue", label="Current Oriented Points")
                ax.legend()
                fig.canvas.draw()
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                frames_map.append(frame_bgr)
                plt.close(fig)
            
        # Switch the current oriented points to previous for the next iteration
        oriented_points1 = oriented_points2
    
    if plot_map:
        if len(frames_map) > 0:
            h, w = frames_map[0].shape[:2]
            file_name = f"odometry_map_{filename_suffix}.mp4"
            writer = cv2.VideoWriter(
                os.path.join(output_dir, file_name),
                cv2.VideoWriter_fourcc(*"mp4v"),
                1.0,
                (w, h),
                True,
            )
            for frame in frames_map:
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
            writer.release()

    return odometry_results, timestamps, used_images


def ablation_study(
    param_grid,
    run_experiment,
    score_column="pos_fault_m",
    score_reducer="mean",
    lower_is_better=True,
    max_runs=None,
    sample_mode="first",
    random_seed=42,
    verbose=True,
    ):
    """
    Run a grid-search-style ablation and return ranked results plus best params.

    Parameters
    ----------
    param_grid : dict[str, list]
        Example: {"k": [300, 500], "r_param": [2.0, 3.0]}
    run_experiment : callable
        Called as run_experiment(**params), must return a pandas DataFrame.
    score_column : str
        Metric column in the returned DataFrame.
    score_reducer : str
        One of: "mean", "median", "max", "p90".
    lower_is_better : bool
        If True, lowest score is ranked best.
    max_runs : int | None
        Maximum number of parameter combinations to evaluate.
        If None, run all combinations.
    sample_mode : str
        "first" uses the first max_runs combinations.
        "random" samples max_runs combinations randomly.
    random_seed : int
        Seed used when sample_mode="random".
    verbose : bool
        If True, prints progress for each parameter combination.
    """
    from itertools import product
    from time import perf_counter

    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dictionary")

    bad_keys = [
        k for k, v in param_grid.items()
        if not isinstance(v, (list, tuple)) or len(v) == 0
    ]
    if bad_keys:
        raise ValueError(
            f"Each entry in param_grid must be a non-empty list/tuple. Invalid keys: {bad_keys}"
        )

    reducers = {
        "mean": lambda s: float(np.mean(s)),
        "median": lambda s: float(np.median(s)),
        "max": lambda s: float(np.max(s)),
        "p90": lambda s: float(np.percentile(s, 90)),
    }
    if score_reducer not in reducers:
        raise ValueError(f"score_reducer must be one of {list(reducers.keys())}")

    keys = list(param_grid.keys())
    all_combos = list(product(*[param_grid[k] for k in keys]))

    if max_runs is not None:
        if not isinstance(max_runs, int) or max_runs <= 0:
            raise ValueError("max_runs must be a positive integer or None")

        max_runs = min(max_runs, len(all_combos))
        if sample_mode == "first":
            combos = all_combos[:max_runs]
        elif sample_mode == "random":
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(len(all_combos), size=max_runs, replace=False)
            combos = [all_combos[i] for i in idx]
        else:
            raise ValueError("sample_mode must be 'first' or 'random'")
    else:
        combos = all_combos

    rows = []

    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        t0 = perf_counter()

        try:
            result_df = run_experiment(**params)
            if not isinstance(result_df, pd.DataFrame):
                raise TypeError("run_experiment must return a pandas DataFrame")
            if score_column not in result_df.columns:
                raise KeyError(f"Missing score column: {score_column}")

            values = result_df[score_column].dropna().to_numpy(dtype=float)
            if values.size == 0:
                score = np.inf if lower_is_better else -np.inf
                status = "empty-score"
            else:
                score = reducers[score_reducer](values)
                status = "ok"
        except Exception as exc:
            score = np.inf if lower_is_better else -np.inf
            status = f"failed: {type(exc).__name__}: {exc}"

        runtime_s = perf_counter() - t0
        rows.append(
            {
                **params,
                "score": float(score),
                "score_column": score_column,
                "score_reducer": score_reducer,
                "runtime_s": float(runtime_s),
                "status": status,
            }
        )

        if verbose:
            print(
                f"[{idx}/{len(combos)}] {params} -> "
                f"score={score:.4f}, status={status}, runtime={runtime_s:.2f}s"
            )

    summary_df = pd.DataFrame(rows).sort_values(
        by=["score", "runtime_s"],
        ascending=[lower_is_better, True],
        ignore_index=True,
    )

    valid = summary_df[summary_df["status"] == "ok"]
    best_params = {k: valid.iloc[0][k] for k in keys} if not valid.empty else {}

    return summary_df, best_params

def run_experiment(n_images, **params):
    # This function should run the odometry pipeline with the given parameters and return a DataFrame with a "pos_fault_m" column.

    odometry_results, timestamps, used_images = main_odometry(
        n_images= n_images,
        preprocessing=params.get("preprocessing", preprocessing),
        k=params.get("k", k),
        z_percentile=params.get("z_percentile", z_percentile),
        r_param=params.get("r_param", r_param),
        f_param=params.get("f_param", f_param),
        cost_function=params.get("cost_function", cost_function),
        ICP=params.get("ICP", ICP),
        window_size=params.get("window_size", window_size),
        motion_compensation_flag=params.get("motion_compensation_flag", motion_compensation_flag),
    )
    return odometry_results

param_grid = {
    "preprocessing": ["normalized_azimuths", "cfar"],
    "k": [12, 25, 40],
    "r_param": [3.0, 5.0, 8.0],
    "f_param": [1.0, 2.0],
    "cost_function": ["p2p", "p2l", "p2d"],
    "system": ["cartesian"],
    "ICP": [True, False],
    "z_percentile": [99.0, 99.5],
    "window_size": [1, 3, 5],
    "motion_compensation_flag": [True, False],
}

n_images = 90

summary, best_params = ablation_study(
    param_grid=param_grid,
    run_experiment=lambda **p: run_experiment(n_images=n_images, **p),
    score_column="pos_fault_m",
    score_reducer="mean",
    lower_is_better=True,
    max_runs=30,
    sample_mode="random",
    random_seed=42,
    verbose=True,
)

display(summary.head(10))
print("Best params:", best_params)

# Save summary to CSV
summary.to_csv("ablation_summary.csv", index=False)