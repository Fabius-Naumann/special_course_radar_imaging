import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime
from pathlib import Path
from itertools import product
from time import perf_counter

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loading import extract_timestamp, load_gps_data, load_radar_images
from descriptors import computing_CFEAR_Features, transform_oriented_points
from data_association import (
    ICP_registration,
    registration_from_oriented_points,
)
from evaluation import (
    calculate_gps_distance,
    calculate_odo_longitude_latitude,
    interpolate_gps_motion,
)


DEFAULT_ODOMETRY_PARAMS = {
    "preprocessing": "normalized_azimuths",
    "k": 25,
    "z_percentile": 99.5,
    "max_distance_percentile": 75,
    "r_param": 5.0,
    "f_param": 1.0,
    "cost_function": "p2p",
    "ICP": False,
    "window_size": 3,
    "motion_compensation_flag": True,
    "every_nth_frame": 1,
    "max_iterations": 10,
}

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
    n_jobs=1,
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
    n_jobs : int
        Number of parallel workers used to evaluate parameter combinations.
        Use 1 for sequential execution, -1 to use all available CPU cores.
    """
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

    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be an integer and not equal to 0")
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")

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

    def evaluate_one(combo):
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
        return {
            **params,
            "score": float(score),
            "score_column": score_column,
            "score_reducer": score_reducer,
            "runtime_s": float(runtime_s),
            "status": status,
        }

    rows = []
    if n_jobs == 1:
        for idx, combo in enumerate(combos, start=1):
            row = evaluate_one(combo)
            rows.append(row)
            if verbose:
                params = {k: row[k] for k in keys}
                print(
                    f"[{idx}/{len(combos)}] {params} -> "
                    f"score={row['score']:.4f}, status={row['status']}, runtime={row['runtime_s']:.2f}s"
                )
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(evaluate_one, combo): combo for combo in combos}
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                if verbose:
                    params = {k: row[k] for k in keys}
                    print(
                        f"[{completed}/{len(combos)}] {params} -> "
                        f"score={row['score']:.4f}, status={row['status']}, runtime={row['runtime_s']:.2f}s"
                    )

    summary_df = pd.DataFrame(rows).sort_values(
        by=["score", "runtime_s"],
        ascending=[lower_is_better, True],
        ignore_index=True,
    )

    valid = summary_df[summary_df["status"] == "ok"]
    best_params = {k: valid.iloc[0][k] for k in keys} if not valid.empty else {}

    return summary_df, best_params

def run_experiment(images, image_files, gps_data, **params):
    # This function should run the odometry pipeline with the given parameters and return a DataFrame with a "pos_fault_m" column.
    config = {**DEFAULT_ODOMETRY_PARAMS, **params}

    odometry_results, timestamps, used_images = main_odometry(
        images=images,
        image_files=image_files,
        gps_data=gps_data,
        preprocessing=config["preprocessing"],
        k=config["k"],
        z_percentile=config["z_percentile"],
        max_distance_percentile=config["max_distance_percentile"],
        r_param=config["r_param"],
        f_param=config["f_param"],
        cost_function=config["cost_function"],
        ICP=config["ICP"],
        window_size=config["window_size"],
        motion_compensation_flag=config["motion_compensation_flag"],
        every_nth_frame=config["every_nth_frame"],
        max_iterations=config["max_iterations"],
    )
    return odometry_results

def filter_lonely_points(oriented_points, eps=0.5, min_samples=2):
    oriented_points = ensure_oriented_points_array(oriented_points)
    if oriented_points.shape[0] == 0 or oriented_points.shape[1] < 2:
        return np.empty((0, max(2, oriented_points.shape[1])), dtype=float)

    xy = oriented_points[:, :2]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = clustering.labels_

    # Keep points that are in clusters (label != -1)
    filtered_points = oriented_points[labels != -1]
    return filtered_points


def ensure_oriented_points_array(oriented_points):
    arr = np.asarray(oriented_points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)

    if arr.ndim == 1:
        if arr.shape[0] < 2:
            return np.empty((0, 2), dtype=float)
        arr = arr.reshape(1, -1)

    return arr

def main_odometry(
    images,
    image_files,
    gps_data,
    preprocessing,
    k,
    z_percentile,
    max_distance_percentile,
    r_param,
    f_param,
    cost_function,
    ICP,
    window_size,
    motion_compensation_flag,
    every_nth_frame=1,
    max_iterations=50,
    print_lines=False,
    smoothing=None,
    clustering=None,
):
    n_images = min(len(images), len(image_files))
    if n_images == 0:
        raise ValueError("No images/image_files provided.")
    if every_nth_frame <= 0:
        raise ValueError("every_nth_frame must be a positive integer.")
    
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

    i = 0

    while len(oriented_points1) == 0:
        if print_lines : print("Computing CFEAR features for the first image...")
        oriented_points1 = computing_CFEAR_Features(images[i], preprocessing, k, z_percentile, max_distance_percentile, [0, 0, 0], r_param, f_param, motion_compensation_flag=False, smoothing=smoothing)
        oriented_points1 = ensure_oriented_points_array(oriented_points1)

        if len(oriented_points1) == 0:
            if print_lines : print("No oriented points extracted from the first image, skipping that image...")
            i += 1
            if i >= n_images:
                raise ValueError("No oriented points could be extracted from any of the images. Please check the data and parameters.")
        else:
            if print_lines : print(f"Successfully extracted {len(oriented_points1)} oriented points from image {i+1}.")

            # Take the GPS data at the corresponding time as initial values for the first frame
            timestamp = extract_timestamp(image_files[i])
            gps_df = interpolate_gps_motion(gps_data, [timestamp])

            # GPS bearing is compass convention (0°=North, clockwise).
            # Convert to math yaw used by the odometry pipeline (0°=+x/East, counterclockwise).
            initial_bearing_compass = float(gps_df["bearing"].values[0])
            if np.isnan(initial_bearing_compass):
                initial_theta_deg = 0.0
                if print_lines : print("Warning: initial GPS bearing is NaN; using 0.0° for initialization.")
            else:
                initial_theta_deg = 135.0 - initial_bearing_compass

            heading_offset_deg = float(initial_theta_deg)

            initial_theta_rad = np.radians(initial_theta_deg)
            R_1_f = np.array([
                [np.cos(initial_theta_rad), -np.sin(initial_theta_rad)],
                [np.sin(initial_theta_rad), np.cos(initial_theta_rad)],
            ], dtype=float)

            # Add that data to the odometry results dataframe
            first_row = {
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
            odometry_results.loc[len(odometry_results)] = first_row
            timestamps.append(timestamp)
            used_images.append(images[i])

            # Filter the oriented points of the first frame to remove lonely points
            if clustering is True:
                oriented_points1 = filter_lonely_points(oriented_points1, eps=r_param, min_samples=2)
                if print_lines : print(f"After filtering lonely points, {len(oriented_points1)} oriented points remain in the first frame.")

            # Always add the first successfully processed frame as the initial keyframe
            keyframes.append(oriented_points1)

            last_kf_latitude = float(gps_df["latitude"].values[0])
            last_kf_longitude = float(gps_df["longitude"].values[0])

            i += 1

    for img in range(i, n_images, every_nth_frame):
        """Compute the features of the current frame""" 
        # Extract the velocity from the DataFrame for motion compensation in the current frame
        velocity = (
            float(odometry_results["velocity_x_m_s"].values[-1]),
            float(odometry_results["velocity_y_m_s"].values[-1]),
            float(odometry_results["velocity_theta_deg_s"].values[-1]),
        )
        if abs(velocity[2]) > 2.0:
            if print_lines : print(f"Warning: high angular velocity {velocity[2]:.2f} deg/s detected in the last odometry result; capping it to 2.0 deg/s")
            velocity = (velocity[0], velocity[1], 0.0)

        # Computing CFEAR features for the current image
        if print_lines : print(f"Processing image {img+1}/{n_images}...")
        oriented_points2 = computing_CFEAR_Features(images[img], preprocessing, k, z_percentile, max_distance_percentile, velocity, r_param, f_param, motion_compensation_flag=motion_compensation_flag, smoothing=smoothing)
        oriented_points2 = ensure_oriented_points_array(oriented_points2)

        if print_lines : print(f"Extracted {len(oriented_points2)} oriented surface points from image {img+1}.")
        if len(oriented_points2) == 0:
            if print_lines : print(f"No oriented points extracted from image {img+1}, skipping registration.")
            continue

        # Initial guess based on last pose and last velocity

        # Compute time difference dt between the current frame and the previous frame
        curr_timestamp = extract_timestamp(image_files[img])
        prev_timestamp = str(odometry_results["timestamp"].values[-1])

        t_prev = datetime.fromisoformat(prev_timestamp)
        t_curr = datetime.fromisoformat(curr_timestamp)

        dt = (t_curr - t_prev).total_seconds()
        
        initial_theta = np.radians(float(odometry_results["rotation_deg"].values[-1]) - heading_offset_deg + (float(velocity[2]) * dt))
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
            start_theta=initial_theta, start_t=initial_t, max_iterations=max_iterations
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
        oriented_points2_T = ensure_oriented_points_array(oriented_points2_T)

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
            # Filter the oriented points of the current frame to remove lonely points before adding as a new keyframe
            if clustering is True:
                oriented_points2_T = filter_lonely_points(oriented_points2_T, eps=r_param, min_samples=2)
                if print_lines : print(f"After filtering lonely points, {len(oriented_points2_T)} oriented points remain in the current frame.")
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

        # Append current frame to odometry results
        current_row = {
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
        odometry_results.loc[len(odometry_results)] = current_row
        timestamps.append(curr_timestamp)
        used_images.append(images[img])
  
        # Switch the current oriented points to previous for the next iteration
        oriented_points1 = oriented_points2


    return odometry_results, timestamps, used_images


param_grid = {
    "preprocessing": ["normalized_azimuths", "cfar"],
    "k": [12, 25, 40],
    "max_distance_percentile": [100],
    "r_param": [3.0, 5.0, 8.0],
    "f_param": [1.0, 2.0],
    "cost_function": ["p2p", "p2l", "p2d"],
    "ICP": [False],
    "z_percentile": [99.0, 99.5],
    "window_size": [1, 3, 5],
    "motion_compensation_flag": [True, False],
    "every_nth_frame": [1, 2, 3],
}

n_images = 120
dataset = 1

PROJECT_DIR = Path(__file__).parent.parent

if __name__ == "__main__":
    if dataset == 1:
        folder_path = PROJECT_DIR / "data" / "_radar_data_b_scan_image"
    elif dataset == 2:
        folder_path = PROJECT_DIR / "data" / "_radar_data_b_scan_image_2"
    else:
        raise ValueError("dataset must be either 1 or 2")

    image_files, images = load_radar_images(
        num_images=n_images,
        correct_black_lines_flag=True,
        folder_path=folder_path,
    )
    gps_data = load_gps_data()

    summary, best_params = ablation_study(
        param_grid=param_grid,
        run_experiment=lambda **p: run_experiment(images=images, image_files=image_files, gps_data=gps_data, **p),
        score_column="pos_fault_m",
        score_reducer="p90",
        lower_is_better=True,
        max_runs=50,
        sample_mode="random",
        random_seed=42,
        verbose=True,
        n_jobs=-1,
    )

    print(summary.head(10))
    print("Best params:", best_params)

    # Save summary to CSV
    summary.to_csv("ablation_summary.csv", index=False)
