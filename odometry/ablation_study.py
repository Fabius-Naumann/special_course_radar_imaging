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
	interpolate_gps_motion,
	rigid_body_transform,
)
from utils import DATA_DIR, RESULTS_DIR
from utils.data_loading import extract_timestamp, load_gps_data, load_radar_images


GPS_TO_RADAR_OFFSET_M = 9.0
GPS_TO_RADAR_OFFSET_ANGLE_DEG = -90.0
RADAR_TO_GPS_OFFSET_ANGLE_DEG = GPS_TO_RADAR_OFFSET_ANGLE_DEG + 180.0


# Baseline setup from main.ipynb.
DEFAULT_ODOMETRY_PARAMS = {
	"preprocessing": "normalized_azimuths",  # "normalized_azimuths", "cfar", None
	"k": 20,
	"max_distance_percentile": 25,
	"r_param": 3.0,
	"f_param": 1.0,
	"n_images": 100,
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
	"preprocessing": ["normalized_azimuths", "cfar"],
	"k": [12, 25, 40],
	"max_distance_percentile": [100],
	"r_param": [3.0, 5.0],
	"f_param": [1.0, 2.0],
	"z_percentile": [99.0, 99.5],
	"cost_function": ["p2p", "p2l", "p2d"],
	"ICP": [False],
	"window_size": [1, 3, 5],
	"motion_compensation_flag": [False, True],
	"iterations": [8],
	"smoothing": [None],
	"filtering": [False],
	"every_nth_frame": [1, 2, 3],
}


def filter_lonely_points(oriented_points, eps=0.5, min_samples=2):
	if len(oriented_points) == 0:
		return np.empty((0, 4), dtype=float)

	xy = oriented_points[:, :2]
	clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
	labels = clustering.labels_
	return oriented_points[labels != -1]


def _rotation_matrix_from_deg(theta_deg):
	theta_rad = np.radians(theta_deg)
	c, s = np.cos(theta_rad), np.sin(theta_rad)
	return np.array([[c, -s], [s, c]], dtype=float)


def main_odometry(
	n_images,
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
	iterations,
	every_nth_frame=1,
	print_lines=False,
	smoothing=None,
	clustering=False,
	dataset=1,
):
	if dataset == 1:
		dataset_dir = DATA_DIR / "_radar_data_b_scan_image"
	elif dataset == 2:
		dataset_dir = DATA_DIR / "_radar_data_b_scan_image_2"
	else:
		raise ValueError("Invalid dataset selection. Please choose 1 or 2.")

	image_file, image = load_radar_images(
		num_images=n_images,
		correct_black_lines_flag=True,
		folder_path=dataset_dir,
	)
	gps_data = load_gps_data()

	if every_nth_frame > 1:
		image_file = image_file[::every_nth_frame]
		image = image[::every_nth_frame]

	if len(image) < 2:
		raise ValueError("Not enough radar frames after sub-sampling to run odometry.")

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
		]
	)

	timestamps = []
	keyframes = []
	used_images = []
	heading_offset_deg = 0.0

	i = 0
	oriented_points1 = np.empty((0, 4), dtype=float)
	R_1_f = np.eye(2, dtype=float)
	t_1_f = np.zeros(2, dtype=float)
	last_kf_latitude = None
	last_kf_longitude = None

	while len(oriented_points1) == 0:
		if i >= len(image):
			raise ValueError(
				"No oriented points could be extracted from any frame. "
				"Please check data and parameters."
			)

		if print_lines:
			print(f"Computing CFEAR features for initial frame {i + 1}...")

		oriented_points1 = computing_CFEAR_Features(
			image[i],
			preprocessing,
			k,
			z_percentile,
			max_distance_percentile,
			[0, 0, 0],
			r_param,
			f_param,
			motion_compensation_flag=False,
			smoothing=smoothing,
		)

		if len(oriented_points1) == 0:
			i += 1
			continue

		timestamp = extract_timestamp(image_file[i])
		gps_df = interpolate_gps_motion(gps_data, [timestamp])

		initial_bearing_compass = float(gps_df["bearing"].values[0])
		if np.isnan(initial_bearing_compass):
			initial_theta_deg = 0.0
			if print_lines:
				print("Warning: initial GPS bearing is NaN; using 0.0 deg.")
		else:
			# Keep notebook conversion exactly for consistency.
			initial_theta_deg = 135.0 - initial_bearing_compass

		heading_offset_deg = float(initial_theta_deg)
		R_1_f = _rotation_matrix_from_deg(initial_theta_deg)

		initial_longitude = float(gps_df["longitude"].values[0])
		initial_latitude = float(gps_df["latitude"].values[0])
		initial_latitude, initial_longitude = rigid_body_transform(
			initial_latitude,
			initial_longitude,
			heading_deg=heading_offset_deg,
			offset_distance_m=GPS_TO_RADAR_OFFSET_M,
			offset_angle_deg=GPS_TO_RADAR_OFFSET_ANGLE_DEG,
		)

		if clustering:
			oriented_points1 = filter_lonely_points(
				oriented_points1,
				eps=r_param,
				min_samples=2,
			)

		first_row = pd.DataFrame(
			[
				{
					"timestamp": timestamp,
					"rotation_deg": float(initial_theta_deg),
					"translation_x_m": 0.0,
					"translation_y_m": 0.0,
					"odometry_longitude": float(initial_longitude),
					"odometry_latitude": float(initial_latitude),
					"estimated_gps_longitude": float(gps_df["longitude"].values[0]),
					"estimated_gps_latitude": float(gps_df["latitude"].values[0]),
					"velocity_x_m_s": 0.0,
					"velocity_y_m_s": 0.0,
					"velocity_theta_deg_s": 0.0,
					"pos_fault_m": 0.0,
				}
			]
		)
		odometry_results = pd.concat([odometry_results, first_row], ignore_index=True)
		timestamps.append(timestamp)
		used_images.append(image[i])

		keyframes.append(oriented_points1)
		last_kf_latitude = float(first_row["odometry_latitude"].values[0])
		last_kf_longitude = float(first_row["odometry_longitude"].values[0])
		i += 1

	for img_idx in range(i, len(image)):
		velocity = (
			float(odometry_results["velocity_x_m_s"].values[-1]),
			float(odometry_results["velocity_y_m_s"].values[-1]),
			float(odometry_results["velocity_theta_deg_s"].values[-1]),
		)
		if abs(velocity[2]) > 2.0:
			velocity = (velocity[0], velocity[1], 0.0)

		oriented_points2 = computing_CFEAR_Features(
			image[img_idx],
			preprocessing,
			k,
			z_percentile,
			max_distance_percentile,
			velocity,
			r_param,
			f_param,
			motion_compensation_flag=motion_compensation_flag,
			smoothing=smoothing,
		)

		if len(oriented_points2) == 0:
			continue

		curr_timestamp = extract_timestamp(image_file[img_idx])
		prev_timestamp = str(odometry_results["timestamp"].values[-1])
		dt = (
			datetime.fromisoformat(curr_timestamp)
			- datetime.fromisoformat(prev_timestamp)
		).total_seconds()

		initial_theta = np.radians(
			float(odometry_results["rotation_deg"].values[-1])
			- heading_offset_deg
			+ (float(velocity[2]) * dt)
		)
		initial_t = np.array(
			[
				float(odometry_results["translation_x_m"].values[-1]),
				float(odometry_results["translation_y_m"].values[-1]),
			],
			dtype=float,
		)
		initial_t += np.array([velocity[0], velocity[1]], dtype=float) * dt

		R_f_kf, t_f_kf, correspondences, _cov = registration_from_oriented_points(
			oriented_points2,
			keyframes,
			cost_function=cost_function,
			return_covariance=True,
			start_theta=initial_theta,
			start_t=initial_t,
			max_iterations=iterations,
		)
		if len(correspondences) == 0:
			continue

		R_f_kf = np.asarray(R_f_kf, dtype=float)
		t_f_kf = np.asarray(t_f_kf, dtype=float)

		theta_rad = np.arctan2(R_f_kf[1, 0], R_f_kf[0, 0])
		theta_deg = float(np.degrees(theta_rad))
		theta_deg = ((theta_deg + heading_offset_deg + 180.0) % 360.0) - 180.0
		theta_rad = np.radians(theta_deg)

		prev_theta_deg = float(odometry_results["rotation_deg"].values[-1])
		prev_theta = np.radians(prev_theta_deg)
		c_prev, s_prev = np.cos(prev_theta), np.sin(prev_theta)
		R_1_prev = np.array([[c_prev, -s_prev], [s_prev, c_prev]], dtype=float)
		t_1_prev = np.array(
			[
				float(odometry_results["translation_x_m"].values[-1]),
				float(odometry_results["translation_y_m"].values[-1]),
			],
			dtype=float,
		)

		if ICP:
			points1 = np.hstack(
				(oriented_points1[:, :2], np.zeros((oriented_points1.shape[0], 1)))
			)
			points2 = np.hstack(
				(oriented_points2[:, :2], np.zeros((oriented_points2.shape[0], 1)))
			)

			R_prev_1 = R_1_prev.T
			t_prev_1 = -t_1_prev @ R_1_prev
			R_f_prev_init = R_prev_1 @ R_1_f
			t_f_prev_init = t_1_f @ R_prev_1.T + t_prev_1

			transform = ICP_registration(
				points2,
				points1,
				R_init=R_f_prev_init,
				t_init=t_f_prev_init,
				distance_threshold=0.2,
			)
			R_f_prev = transform[:2, :2]
			t_f_prev = transform[:2, 3]

			R_1_f = R_1_prev @ R_f_prev
			t_1_f = t_f_prev @ R_1_prev.T + t_1_prev

			theta_rad = np.arctan2(R_1_f[1, 0], R_1_f[0, 0])
			theta_deg = float(np.degrees(theta_rad))
			theta_deg = ((theta_deg + heading_offset_deg + 180.0) % 360.0) - 180.0
			theta_rad = np.radians(theta_deg)
		else:
			R_1_f = R_f_kf
			t_1_f = t_f_kf

		oriented_points2_t = transform_oriented_points(oriented_points2, R_1_f, t_1_f)

		last_odo_longitude = float(odometry_results["odometry_longitude"].values[-1])
		last_odo_latitude = float(odometry_results["odometry_latitude"].values[-1])
		delta_global = t_1_f - np.array(
			[
				float(odometry_results["translation_x_m"].values[-1]),
				float(odometry_results["translation_y_m"].values[-1]),
			],
			dtype=float,
		)
		odometry_latitude, odometry_longitude = calculate_odo_longitude_latitude(
			last_odo_latitude,
			last_odo_longitude,
			delta_global,
			theta_rad,
		)
		estimated_gps_latitude, estimated_gps_longitude = rigid_body_transform(
			odometry_latitude,
			odometry_longitude,
			theta_deg,
			offset_distance_m=GPS_TO_RADAR_OFFSET_M,
			offset_angle_deg=RADAR_TO_GPS_OFFSET_ANGLE_DEG,
		)

		gps_entry_kf = {
			"latitude": float(last_kf_latitude),
			"longitude": float(last_kf_longitude),
		}
		gps_entry_curr = {
			"latitude": float(odometry_latitude),
			"longitude": float(odometry_longitude),
		}
		keyframe_distance_m = calculate_gps_distance(gps_entry_kf, gps_entry_curr)

		if keyframe_distance_m > 1.0:
			if clustering:
				oriented_points2_t = filter_lonely_points(
					oriented_points2_t,
					eps=r_param,
					min_samples=2,
				)
			if len(oriented_points2_t) > 0:
				keyframes.append(oriented_points2_t)
				last_kf_latitude = float(odometry_latitude)
				last_kf_longitude = float(odometry_longitude)
				if len(keyframes) > window_size:
					keyframes.pop(0)

		prev_pos = np.array(
			[
				float(odometry_results["translation_x_m"].values[-1]),
				float(odometry_results["translation_y_m"].values[-1]),
			],
			dtype=float,
		)
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

		gps_df = interpolate_gps_motion(gps_data, [curr_timestamp])
		gt_latitude = gps_df["latitude"].values[0]
		gt_longitude = gps_df["longitude"].values[0]
		position_error = calculate_gps_distance(
			{"latitude": float(gt_latitude), "longitude": float(gt_longitude)},
			{
				"latitude": float(estimated_gps_latitude),
				"longitude": float(estimated_gps_longitude),
			},
		)

		current_row = pd.DataFrame(
			[
				{
					"timestamp": curr_timestamp,
					"rotation_deg": float(theta_deg),
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
				}
			]
		)
		odometry_results = pd.concat([odometry_results, current_row], ignore_index=True)
		timestamps.append(curr_timestamp)
		used_images.append(image[img_idx])
		oriented_points1 = oriented_points2

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
	if reducer in {"p95", "percentile95", "q95"}:
		return float(np.percentile(values, 95))
	raise ValueError("Unsupported reducer. Use one of: mean, median, max, p95")


def score_odometry_results(odometry_results, score_column="pos_fault_m", reducer="mean"):
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


def run_single_experiment(base_params, overrides, score_column="pos_fault_m", score_reducer="mean"):
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
		)
		score = score_odometry_results(
			odometry_results,
			score_column=score_column,
			reducer=score_reducer,
		)
		status = "ok"
		n_rows = int(len(odometry_results))
		n_fail_rows = int(np.sum(~np.isfinite(odometry_results[score_column].astype(float).values)))
	except Exception as exc:  # noqa: BLE001
		score = np.inf
		status = f"failed: {type(exc).__name__}: {exc}"
		n_rows = 0
		n_fail_rows = 0

	runtime_s = float(time.perf_counter() - start_time)
	result = {
		**params,
		"score": float(score),
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
	score_reducer="mean",
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
				f"    -> status={row['status']} score={row['score']:.6f} "
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
					f"status={row['status']} score={row['score']:.6f} "
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
		score_reducer="max",
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
