import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# Add project root to sys.path when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from data_association import ICP_registration, registration_from_oriented_points
from descriptors import computing_CFEAR_Features, transform_oriented_points
from evaluation import (
	calculate_gps_distance,
	calculate_odo_longitude_latitude,
	interpolate_gps_motion
)
from utils import DATA_DIR, RESULTS_DIR
from utils.data_loading import extract_timestamp, load_gps_data, load_radar_images


GPS_TO_RADAR_OFFSET_M = 9.0
GPS_TO_RADAR_OFFSET_ANGLE_DEG = 16.0


# Baseline setup from main.ipynb.
DEFAULT_ODOMETRY_PARAMS = {
	"preprocessing": "normalized_azimuths",  # "normalized_azimuths", "cfar", None
	"k": 20,
	"max_distance_percentile": 25,
	"r_param": 3.0,
	"f_param": 1.0,
	"n_images": 251,
	"stop_if_pos_fault_gt_m": 50.0,
	"z_percentile": 99.0,
	"cost_function": "p2p",  # "p2p", "p2l", "p2d"
	"ICP": False,
	"window_size": 3,
	"motion_compensation_flag": True,
	"iterations": 8,
	"smoothing": None,  # "gaussian", "symmetric", None
	"filtering": False,
	"every_nth_frame": 1,
	"dataset": 1,
}


# Start with a compact grid and expand as needed.
PARAM_GRID = {
	"preprocessing": [None, "normalized_azimuths", "cfar"],
	"k": [40],
	"max_distance_percentile": [25, 50, 75, 100],
	"r_param": [5.0],
	"f_param": [2.0],
	"z_percentile": [99.5],
	"cost_function": ["p2p", "p2l"],
	"ICP": [True, False],
	"window_size": [3],
	"motion_compensation_flag": [False],
	"iterations": [8],
	"smoothing": [None, "gaussian", "symmetric"],
	"filtering": [True, False],
	"every_nth_frame": [2],
}



def filter_lonely_points(oriented_points, eps=0.5, min_samples=2):
    if len(oriented_points) == 0:
        return np.array([])

    xy = oriented_points[:, :2]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = clustering.labels_

    # Keep points that are in clusters (label != -1)
    filtered_points = oriented_points[labels != -1]
    return filtered_points

def transform_velocity(velocity_global, heading_deg, gps_to_radar_offset_m=GPS_TO_RADAR_OFFSET_M, gps_to_radar_offset_angle_deg=GPS_TO_RADAR_OFFSET_ANGLE_DEG):
    if velocity_global is None or len(velocity_global) != 3:
        return np.array([0.0, 0.0, 0.0])

    v_x_glob, v_y_glob, v_theta_deg = velocity_global
    v_radar_glob = np.array([v_x_glob, v_y_glob])

    # Convert angles to radians
    heading_rad = np.radians(heading_deg)
    v_theta_rad = np.radians(v_theta_deg)
    offset_angle_rad = np.radians(gps_to_radar_offset_angle_deg)

    # 1. Un-rotate the radar's global velocity to the LOCAL frame
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    R_inv = np.array([
        [c, s],
        [-s, c]
    ])
    v_radar_local = R_inv @ v_radar_glob

    # 2. Compute the GPS lever-arm position in the LOCAL frame
    rx_local = gps_to_radar_offset_m * np.cos(offset_angle_rad)
    ry_local = gps_to_radar_offset_m * np.sin(offset_angle_rad)

    # 3. Compute lever-arm velocity entirely in LOCAL frame (omega x r_local)
    v_lever_local = np.array([-v_theta_rad * ry_local, v_theta_rad * rx_local])

    # 4. Add them to get the total GPS velocity in the LOCAL frame!
    v_gps_local = v_radar_local + v_lever_local

    return np.array([v_gps_local[0], v_gps_local[1], v_theta_deg])

def estimate_gps(last_longitude, last_latitude, velocity_local, heading, dt):
    # 1. Calculate local translation step [dx_local, dy_local]
    step_local = np.array([velocity_local[0] * dt, velocity_local[1] * dt])
    heading_rad = np.radians(heading)

    # 2. Rotate the local step BACK to the global frame
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    R_fwd = np.array([
        [c, -s],
        [s, c]
    ])
    step_global = R_fwd @ step_local

    # 3. Pass GLOBAL step to calculate_odo_longitude_latitude 
    gps_latitude, gps_longitude = calculate_odo_longitude_latitude(
        start_lat=last_latitude, 
        start_lon=last_longitude, 
        translation_m=step_global, 
        rotation_rad=heading_rad
    )
    
    return gps_longitude, gps_latitude

def main_odometry(n_images, preprocessing, k, z_percentile, max_distance_percentile, r_param, f_param, cost_function, ICP, window_size, motion_compensation_flag, iterations, every_nth_frame=1, print_lines=False, plot_map=False, smoothing=None, clustering=None, dataset=1, stop_if_pos_fault_gt_m=50.0):
    if dataset == 1:
        dataset_dir = DATA_DIR / "_radar_data_b_scan_image"
    elif dataset == 2:
        dataset_dir = DATA_DIR / "_radar_data_b_scan_image_2"
    else:
        raise ValueError("Invalid dataset selection. Please choose 1 or 2.")

    image_file, image = load_radar_images(num_images=n_images, correct_black_lines_flag=True, folder_path=dataset_dir)
    gps_data = load_gps_data()

    if every_nth_frame > 1:
        image_file = image_file[::every_nth_frame]
        image = image[::every_nth_frame]

    odometry_results = pd.DataFrame(
        columns=[
            "timestamp",
            "rotation_deg",
            "translation_x_m",
            "translation_y_m",
            "odometry_longitude",
            "odometry_latitude",
            "estimated_gps_longitude",
            "estimated_gps_latitude",
            "velocity_x_m_s",
            "velocity_y_m_s",
            "velocity_theta_deg_s",
            "pos_fault_m",
            "uncomp_pos_fault_m"
        ]
    )

    timestamps = []
    keyframes = []
    last_kf_longitude = None
    last_kf_latitude = None

    oriented_points1 = []
    R_1_f = np.eye(2)
    t_1_f = np.zeros(2, dtype=float)
    used_images = []
    heading_offset_deg = 0.0
    frames_map = []

    i = 0

    while len(oriented_points1) == 0:
        if i >= len(image):
            raise ValueError("No oriented points could be extracted from any image.")

        if print_lines : print("Computing CFEAR features for the first image...")
        oriented_points1 = computing_CFEAR_Features(image[i], preprocessing, k, z_percentile, max_distance_percentile, [0, 0, 0], r_param, f_param, motion_compensation_flag=False, smoothing=smoothing)

        if len(oriented_points1) == 0:
            if print_lines : print("No oriented points extracted from the first image, skipping that image...")
            i += 1
            if i >= len(image):
                raise ValueError("No oriented points could be extracted from any of the images. Please check the data and parameters.")
        else:
            if print_lines : print(f"Successfully extracted {len(oriented_points1)} oriented points from image {i+1}.")

            timestamp = extract_timestamp(image_file[i])
            gps_df = interpolate_gps_motion(gps_data, [timestamp])

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

            initial_longitude = float(gps_df["longitude"].values[0])
            initial_latitude = float(gps_df["latitude"].values[0])

            first_row = pd.DataFrame([
                {
                    "timestamp": timestamp,
                    "rotation_deg": float(initial_theta_deg),
                    "translation_x_m": 0.0,
                    "translation_y_m": 0.0,
                    "odometry_longitude": initial_longitude,
                    "odometry_latitude": initial_latitude,
                    "estimated_gps_longitude": float(gps_df["longitude"].values[0]),
                    "estimated_gps_latitude": float(gps_df["latitude"].values[0]),
                    "velocity_x_m_s": 0.0,
                    "velocity_y_m_s": 0.0,
                    "velocity_theta_deg_s": 0.0,
                    "pos_fault_m": 0.0,
                    "uncomp_pos_fault_m": 0.0,
                }
            ])
            odometry_results = pd.concat([odometry_results, first_row], ignore_index=True)
            timestamps.append(timestamp)
            used_images.append(image[i])

            if clustering == True:
                oriented_points1 = filter_lonely_points(oriented_points1, eps=r_param, min_samples=2)
                if print_lines : print(f"After filtering lonely points, {len(oriented_points1)} oriented points remain in the first frame.")

            keyframes.append(oriented_points1)

            last_kf_latitude = float(first_row["odometry_latitude"].values[0])
            last_kf_longitude = float(first_row["odometry_longitude"].values[0])

            i += 1

    for img in range(i, len(image)):
        velocity = (
            float(odometry_results["velocity_x_m_s"].values[-1]),
            float(odometry_results["velocity_y_m_s"].values[-1]),
            float(odometry_results["velocity_theta_deg_s"].values[-1]),
        )

        if print_lines : print(f"Processing image {(every_nth_frame*img)+1}/{n_images}...")
        oriented_points2 = computing_CFEAR_Features(image[img], preprocessing, k, z_percentile, max_distance_percentile, velocity, r_param, f_param, motion_compensation_flag=motion_compensation_flag, smoothing=smoothing)

        if print_lines : print(f"Extracted {len(oriented_points2)} oriented surface points from image {(every_nth_frame*img)+1}.")
        if len(oriented_points2) == 0:
            if print_lines : print(f"No oriented points extracted from image {(every_nth_frame*img)+1}, skipping registration.")
            continue

        from datetime import datetime
        curr_timestamp = extract_timestamp(image_file[img])
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
            points1 = np.hstack((points1, np.zeros((points1.shape[0], 1))))
            points2 = np.hstack((points2, np.zeros((points2.shape[0], 1))))

            R_prev_1 = R_1_prev.T
            t_prev_1 = -t_1_prev @ R_1_prev
            R_f_prev_init = R_prev_1 @ R_1_f
            t_f_prev_init = t_1_f @ R_prev_1.T + t_prev_1

            transform = ICP_registration(points2, points1, R_init=R_f_prev_init, t_init=t_f_prev_init, distance_threshold=0.2)
            R_f_prev = transform[:2, :2]
            t_f_prev = transform[:2, 3]

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

        oriented_points2_T = transform_oriented_points(oriented_points2, R_1_f, t_1_f)

        last_odo_longitude = float(odometry_results["odometry_longitude"].values[-1])
        last_odo_latitude = float(odometry_results["odometry_latitude"].values[-1])
        delta_global = t_1_f - np.array([
            float(odometry_results["translation_x_m"].values[-1]),
            float(odometry_results["translation_y_m"].values[-1]),
        ], dtype=float)
        
        odometry_latitude, odometry_longitude = calculate_odo_longitude_latitude(
            start_lat=last_odo_latitude,
            start_lon=last_odo_longitude,
            translation_m=delta_global,
            rotation_rad=theta_rad
        )

        gps_entry_kf = {"latitude": float(last_kf_latitude), "longitude": float(last_kf_longitude)}
        gps_entry_curr = {"latitude": float(odometry_latitude), "longitude": float(odometry_longitude)}
        keyframe_distance_m = calculate_gps_distance(gps_entry_kf, gps_entry_curr)

        if keyframe_distance_m > 1.0:
            if clustering == True:
                oriented_points2_T = filter_lonely_points(oriented_points2_T, eps=r_param, min_samples=2)
                if print_lines : print(f"After filtering lonely points, {len(oriented_points2_T)} oriented points remain in the current frame.")
            keyframes.append(oriented_points2_T)
            last_kf_latitude = float(odometry_latitude)
            last_kf_longitude = float(odometry_longitude)
            if print_lines : print(f"Added image {img+1} as a new keyframe.")
            if len(keyframes) > window_size:
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

        last_est_gps_longitude = float(odometry_results["estimated_gps_longitude"].values[-1])
        last_est_gps_latitude = float(odometry_results["estimated_gps_latitude"].values[-1])
        
        velocity_gps = transform_velocity(velocity, heading_deg=theta_deg, gps_to_radar_offset_m=GPS_TO_RADAR_OFFSET_M, gps_to_radar_offset_angle_deg=GPS_TO_RADAR_OFFSET_ANGLE_DEG)
        
        estimated_gps_longitude, estimated_gps_latitude = estimate_gps(
            last_longitude=last_est_gps_longitude, 
            last_latitude=last_est_gps_latitude, 
            velocity_local=velocity_gps, 
            heading=theta_deg, 
            dt=dt
        )

        gps_df = interpolate_gps_motion(gps_data, [curr_timestamp])
        gt_latitude = gps_df["latitude"].values[0]
        gt_longitude = gps_df["longitude"].values[0]
        position_error = calculate_gps_distance(
            {"latitude": float(gt_latitude), "longitude": float(gt_longitude)},
            {"latitude": float(estimated_gps_latitude), "longitude": float(estimated_gps_longitude)},
        )
        uncomp_position_error = calculate_gps_distance(
            {"latitude": float(gt_latitude), "longitude": float(gt_longitude)},
            {"latitude": float(odometry_latitude), "longitude": float(odometry_longitude)},
        )
        if print_lines :
            print(f"GPS Ground Truth for current timestamp: latitude: {gt_latitude:.6f}, longitude: {gt_longitude:.6f}")
            print(f"Estimated GPS position: latitude: {estimated_gps_latitude:.6f}, longitude: {estimated_gps_longitude:.6f}")
            print(f"Position error compared to GPS: {position_error:.2f} m")
            print(f"Uncompensated error compared to GPS: {uncomp_position_error:.2f} m")
            print("-" * 50)

        current_row = pd.DataFrame([
            {
                "timestamp": curr_timestamp,
                "rotation_deg": theta_deg,
                "translation_x_m": float(t_1_f[0]),
                "translation_y_m": float(t_1_f[1]),
                "odometry_longitude": float(odometry_longitude),
                "odometry_latitude": float(odometry_latitude),
                "estimated_gps_longitude": float(estimated_gps_longitude),
                "estimated_gps_latitude": float(estimated_gps_latitude),
                "velocity_x_m_s": float(velocity[0]),
                "velocity_y_m_s": float(velocity[1]),
                "velocity_theta_deg_s": float(velocity[2]),
                "pos_fault_m": float(position_error),
                "uncomp_pos_fault_m": float(uncomp_position_error),
            }
        ])
        odometry_results = pd.concat([odometry_results, current_row], ignore_index=True)
        timestamps.append(curr_timestamp)
        used_images.append(image[img])

        if plot_map:
            import cv2
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f"Oriented Points Map - Image {img+1}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_xlim(-400, 400)
            ax.set_ylim(-400, 400)
            ax.set_aspect('equal')

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
            
        oriented_points1 = oriented_points2
    
        if stop_if_pos_fault_gt_m is not None and float(position_error) > float(stop_if_pos_fault_gt_m):
            if print_lines:
                print(f"Stopping run early: pos_fault_m={position_error:.3f} exceeded {float(stop_if_pos_fault_gt_m):.3f} m")
            break

    if plot_map:
        import os
        import cv2
        if len(frames_map) > 0:
            h, w = frames_map[0].shape[:2]
            file_name = f"odometry_map.mp4"
            writer = cv2.VideoWriter(
                os.path.join(".", file_name),
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



def _safe_reducer(values, reducer="mean"):
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]
	if values.size == 0:
		return np.inf

	reducer = str(reducer).lower()
	if reducer == "mean":
		return float(np.mean(values))
	if reducer == "median":
		return float(np.median(values))
	if reducer == "max":
		return float(np.max(values))
	if reducer in {"rmse", "ate_rmse", "root_mean_square_error"}:
		# Translational ATE RMSE from per-frame absolute trajectory errors (meters).
		return float(np.sqrt(np.mean(np.square(values))))
	if reducer in {"p95", "percentile95", "q95"}:
		return float(np.percentile(values, 95))
	raise ValueError(
		"Unsupported reducer. Use one of: mean, median, max, rmse, p95"
	)


def score_odometry_results(odometry_results, score_column="pos_fault_m", reducer="rmse"):
	if odometry_results is None or len(odometry_results) == 0:
		return np.inf
	if score_column not in odometry_results.columns:
		raise KeyError(f"'{score_column}' is not present in odometry results")
	return _safe_reducer(odometry_results[score_column].values, reducer=reducer)


def build_experiments(param_grid, max_experiments=None, random_seed=42):
	keys = list(param_grid.keys())
	values = [param_grid[k] for k in keys]
	combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
	if max_experiments is not None and max_experiments < len(combinations):
		rng = np.random.default_rng(seed=random_seed)
		indices = rng.choice(len(combinations), size=max_experiments, replace=False)
		combinations = [combinations[int(i)] for i in indices]
	return combinations


def run_single_experiment(base_params, overrides, score_column="pos_fault_m", score_reducer="rmse"):
	params = dict(base_params)
	params.update(overrides)

	start_time = time.perf_counter()
	try:
		odometry_results, _, _ = main_odometry(
			n_images=params["n_images"],
			preprocessing=params["preprocessing"],
			k=params["k"],
			z_percentile=params["z_percentile"],
			max_distance_percentile=params["max_distance_percentile"],
			r_param=params["r_param"],
			f_param=params["f_param"],
			cost_function=params["cost_function"],
			ICP=params["ICP"],
			window_size=params["window_size"],
			motion_compensation_flag=params["motion_compensation_flag"],
			iterations=params["iterations"],
			every_nth_frame=params["every_nth_frame"],
			print_lines=False,
			smoothing=params["smoothing"],
			clustering=params["filtering"],
			dataset=params["dataset"],
			stop_if_pos_fault_gt_m=params.get("stop_if_pos_fault_gt_m", 50.0),
		)
		score = score_odometry_results(
			odometry_results,
			score_column=score_column,
			reducer=score_reducer,
		)
		uncomp_score = score_odometry_results(
			odometry_results,
			score_column="uncomp_pos_fault_m",
			reducer=score_reducer,
		)
		status = "ok"
		if score_column == "pos_fault_m":
			max_pos_fault = float(
				np.nanmax(pd.to_numeric(odometry_results["pos_fault_m"], errors="coerce"))
			)
			stop_threshold = params.get("stop_if_pos_fault_gt_m", 50.0)
			if stop_threshold is not None and np.isfinite(max_pos_fault):
				if max_pos_fault > float(stop_threshold):
					status = f"stopped: pos_fault_m>{float(stop_threshold):.1f}"
					score = float(np.inf)
					uncomp_score = float(np.inf)
		n_rows = int(len(odometry_results))
		n_fail_rows = int(np.sum(~np.isfinite(odometry_results[score_column].astype(float).values)))
	except Exception as exc:  # noqa: BLE001
		score = np.inf
		uncomp_score = np.inf
		status = f"failed: {type(exc).__name__}: {exc}"
		n_rows = 0
		n_fail_rows = 0

	runtime_s = float(time.perf_counter() - start_time)
	result = {
		**params,
		"score": float(score),
		"uncomp_score": float(uncomp_score),
		"score_column": score_column,
		"score_reducer": score_reducer,
		"n_rows": n_rows,
		"n_fail_rows": n_fail_rows,
		"runtime_s": runtime_s,
		"status": status,
	}
	return result


def _run_single_experiment_from_payload(payload):
	return run_single_experiment(
		base_params=payload["base_params"],
		overrides=payload["overrides"],
		score_column=payload["score_column"],
		score_reducer=payload["score_reducer"],
	)


def run_ablation(
	base_params,
	param_grid,
	output_dir,
	score_column="pos_fault_m",
	score_reducer="rmse",
	max_experiments=None,
	random_seed=42,
	n_jobs=1,
):
	experiments = build_experiments(
		param_grid,
		max_experiments=max_experiments,
		random_seed=random_seed,
	)
	n_total = len(experiments)
	if n_total == 0:
		raise ValueError("No experiments generated. Check PARAM_GRID.")

	if n_jobs is None or int(n_jobs) <= 0:
		n_jobs = os.cpu_count() or 1
	else:
		n_jobs = int(n_jobs)
	n_jobs = max(1, min(n_jobs, n_total))

	rows = []
	print(f"Running {n_total} ablation experiments with n_jobs={n_jobs}...")

	if n_jobs == 1:
		for idx, overrides in enumerate(experiments, start=1):
			print(f"[{idx}/{n_total}] {overrides}")
			row = run_single_experiment(
				base_params=base_params,
				overrides=overrides,
				score_column=score_column,
				score_reducer=score_reducer,
			)
			rows.append(row)
			print(
				f"    -> status={row['status']} score={row['score']:.6f} uncomp_score={row['uncomp_score']:.6f} "
				f"runtime={row['runtime_s']:.2f}s"
			)
	else:
		payloads = [
			{
				"base_params": base_params,
				"overrides": overrides,
				"score_column": score_column,
				"score_reducer": score_reducer,
			}
			for overrides in experiments
		]

		with ProcessPoolExecutor(max_workers=n_jobs) as executor:
			future_to_overrides = {
				executor.submit(_run_single_experiment_from_payload, payload): payload["overrides"]
				for payload in payloads
			}

			completed = 0
			for future in as_completed(future_to_overrides):
				overrides = future_to_overrides[future]
				completed += 1
				try:
					row = future.result()
				except Exception as exc:  # noqa: BLE001
					row = {
						**base_params,
						**overrides,
						"score": float(np.inf),
						"uncomp_score": float(np.inf),
						"score_column": score_column,
						"score_reducer": score_reducer,
						"n_rows": 0,
						"n_fail_rows": 0,
						"runtime_s": 0.0,
						"status": f"failed: {type(exc).__name__}: {exc}",
					}
				rows.append(row)
				print(
					f"[{completed}/{n_total}] {overrides} -> "
					f"status={row['status']} score={row['score']:.6f} uncomp_score={row['uncomp_score']:.6f} "
					f"runtime={row['runtime_s']:.2f}s"
				)

	df = pd.DataFrame(rows)
	df = df.sort_values(by=["score", "runtime_s"], ascending=[True, True]).reset_index(drop=True)

	output_dir.mkdir(parents=True, exist_ok=True)
	summary_csv_path = output_dir / "ablation_summary.csv"
	summary_json_path = output_dir / "ablation_summary.json"

	df.to_csv(summary_csv_path, index=False)
	df.to_json(summary_json_path, orient="records", indent=2)

	ok_df = df[df["status"] == "ok"]
	best_params = {}
	if len(ok_df) > 0:
		best_row = ok_df.iloc[0].to_dict()
		best_params = {
			k: best_row[k]
			for k in base_params.keys()
			if k in best_row
		}

	best_params_path = output_dir / "ablation_best_params.json"
	with open(best_params_path, "w", encoding="utf-8") as f:
		json.dump(best_params, f, indent=2)

	return df, summary_csv_path, summary_json_path, best_params_path


if __name__ == "__main__":
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	output_dir = RESULTS_DIR / "odometry" / timestamp

	# Keep runtime manageable for a first sweep; set None for full grid.
	max_experiments = 500
	n_jobs = 8

	results_df, summary_csv, summary_json, best_params_json = run_ablation(
		base_params=DEFAULT_ODOMETRY_PARAMS,
		param_grid=PARAM_GRID,
		output_dir=output_dir,
		score_column="pos_fault_m",
		score_reducer="rmse",
		max_experiments=max_experiments,
		random_seed=42,
		n_jobs=n_jobs,
	)

	print("\nAblation finished.")
	print(f"Summary CSV:  {summary_csv}")
	print(f"Summary JSON: {summary_json}")
	print(f"Best params:  {best_params_json}")

	if len(results_df) > 0:
		print("\nTop 5 runs:")
		print(results_df.head(5).to_string(index=False))
