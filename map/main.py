import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from map.shore_detect import cluster_shoreline_dbscan, extract_shoreline  # noqa: E402
from map.shoreline_registration import (  # noqa: E402
    accumulate_shoreline_window,
    build_coastline_map,
    pose_from_geodetic,
    register_shoreline_to_map,
)
from utils import RESULTS_DIR  # noqa: E402
from utils.data_loading import (  # noqa: E402
    FOLDER_PATH,
    correct_black_lines,
    extract_timestamp,
    interpolate_gps_pose,
    load_gps_data,
    offset_latlon_by_local_offsets,
    polar_indices_to_cartesian,
)
from utils.visualisation import (  # noqa: E402
    plot_radar_overlay_on_osm_static,
    plot_shoreline_extraction,
    plot_shoreline_registration_result,
)

GPS_TO_RADAR_OFFSET_M = -9
GPS_TO_RADAR_LATERAL_OFFSET_M = 0.0
HEADING_CORRECTION_DEG = 0.0
RADAR_MAX_RANGE_M = 2000.0
AZIMUTH_CLOCKWISE = True


def parse_args():
    parser = argparse.ArgumentParser(description="Batch shoreline extraction over radar images.")
    parser.add_argument("--process-name", type=str, default="shoreline_batch_test", help="Name of this process group.")
    parser.add_argument("--start-image", type=int, default=0, help="Start index in sorted image list (0-based).")
    parser.add_argument("--max-images", type=int, default=25, help="Maximum number of images to process.")
    parser.add_argument("--step", type=int, default=1, help="Take every N-th image (e.g., 2 = every other image).")
    parser.add_argument(
        "--correct-black-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply black-line correction before detection.",
    )
    parser.add_argument("--grid-size", type=int, default=4096, help="Grid size used for Cartesian mask conversion.")
    parser.add_argument("--min-cluster-area", type=int, default=4, help="Minimum connected component area.")
    parser.add_argument("--morph-open-k", type=int, default=3, help="Morphological opening kernel size.")
    parser.add_argument("--morph-close-k", type=int, default=5, help="Morphological closing kernel size.")
    parser.add_argument(
        "--register-shoreline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run robust shoreline-to-map registration on a short frame window.",
    )
    parser.add_argument("--window-size", type=int, default=5, help="Number of frames in the temporal shoreline window.")
    parser.add_argument(
        "--window-min-persistence",
        type=int,
        default=2,
        help="Minimum number of frames a shoreline cell must persist in to be kept.",
    )
    parser.add_argument(
        "--aggregation-cell-size-m",
        type=float,
        default=4.0,
        help="Anchor-frame grid cell size used during temporal accumulation.",
    )
    parser.add_argument(
        "--shoreline-min-segment-points",
        type=int,
        default=5,
        help="Minimum number of points for a shoreline segment to count as reliable.",
    )
    parser.add_argument(
        "--shoreline-min-segment-length-m",
        type=float,
        default=20.0,
        help="Minimum segment length in meters for a shoreline segment to count as reliable.",
    )
    parser.add_argument(
        "--registration-radius-m",
        type=float,
        default=1500.0,
        help="Radius around the prior position used when building the local coastline map.",
    )
    parser.add_argument(
        "--coastline-sample-step-m",
        type=float,
        default=8.0,
        help="Sampling step used to discretize the vector coastline map.",
    )
    parser.add_argument(
        "--translation-search-m",
        type=float,
        default=150.0,
        help="Translation search radius around the prior during coarse registration.",
    )
    parser.add_argument(
        "--translation-step-m",
        type=float,
        default=15.0,
        help="Translation step size during coarse registration.",
    )
    parser.add_argument(
        "--rotation-search-deg",
        type=float,
        default=180.0,
        help="Yaw search range around the prior heading during coarse registration.",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=10.0,
        help="Yaw step size during coarse registration.",
    )
    parser.add_argument(
        "--coastline-geojson",
        type=str,
        default=None,
        help="Optional local GeoJSON/JSON coastline file. If omitted, the code will try a cache or Overpass fetch.",
    )
    parser.add_argument(
        "--coastline-cache-dir",
        type=str,
        default=None,
        help="Optional directory for cached fetched coastline GeoJSON files.",
    )
    return parser.parse_args()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return cleaned or "shoreline_batch_test"


def _list_sorted_image_files(folder: Path) -> list[Path]:
    return sorted([path for path in folder.iterdir() if path.suffix.lower() == ".png"], key=lambda path: path.name)


def _select_indices(total_images: int, start_image: int, max_images: int, step: int) -> list[int]:
    if total_images == 0:
        return []

    start = max(start_image, 0)
    stride = max(step, 1)
    if start >= total_images:
        return []

    selected = list(range(start, total_images, stride))
    return selected[: max(max_images, 0)]


def _window_indices(anchor_index: int, total_images: int, window_size: int) -> list[int]:
    half_window = max(int(window_size) // 2, 0)
    start = max(0, int(anchor_index) - half_window)
    end = min(total_images, int(anchor_index) + half_window + 1)
    return list(range(start, end))


def _coastline_cache_path(cache_dir: str | None, latitude: float, longitude: float, radius_m: float) -> Path | None:
    if not cache_dir:
        return None
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    filename = f"shoreline_{latitude:.5f}_{longitude:.5f}_{int(radius_m)}m.geojson"
    return cache_root / filename


def main():
    args = parse_args()

    image_files = _list_sorted_image_files(FOLDER_PATH)
    total_available = len(image_files)
    selected_indices = _select_indices(total_available, args.start_image, args.max_images, args.step)

    process_name = _safe_name(args.process_name)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_root = RESULTS_DIR / "shore_detect_runs" / process_name / run_timestamp
    image_dir = run_root / "images"
    points_dir = run_root / "shoreline_points"
    image_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "process_name": process_name,
        "start_image": args.start_image,
        "max_images": args.max_images,
        "step": args.step,
        "grid_size": args.grid_size,
        "min_cluster_area": args.min_cluster_area,
        "morph_open_k": args.morph_open_k,
        "morph_close_k": args.morph_close_k,
        "heading_correction_deg": HEADING_CORRECTION_DEG,
        "gps_lateral_offset_m": GPS_TO_RADAR_LATERAL_OFFSET_M,
        "gps_to_radar_forward_offset_m": GPS_TO_RADAR_OFFSET_M,
        "azimuth_clockwise": AZIMUTH_CLOCKWISE,
        "correct_black_lines": args.correct_black_lines,
        "total_available_images": total_available,
        "selected_indices": selected_indices,
        "register_shoreline": args.register_shoreline,
        "window_size": args.window_size,
        "window_min_persistence": args.window_min_persistence,
        "aggregation_cell_size_m": args.aggregation_cell_size_m,
        "shoreline_min_segment_points": args.shoreline_min_segment_points,
        "shoreline_min_segment_length_m": args.shoreline_min_segment_length_m,
        "registration_radius_m": args.registration_radius_m,
        "coastline_sample_step_m": args.coastline_sample_step_m,
        "translation_search_m": args.translation_search_m,
        "translation_step_m": args.translation_step_m,
        "rotation_search_deg": args.rotation_search_deg,
        "rotation_step_deg": args.rotation_step_deg,
        "coastline_geojson": args.coastline_geojson,
        "coastline_cache_dir": args.coastline_cache_dir,
    }

    with (run_root / "run_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, indent=2)

    print(f"Run directory: {run_root}")
    print(f"Selected {len(selected_indices)} image(s) out of {total_available} available.")

    gps_data = load_gps_data()
    coastline_map_cache = {}

    summary_rows = []
    for image_index in selected_indices:
        image_path = image_files[image_index]
        img = plt.imread(image_path)
        if args.correct_black_lines:
            img = correct_black_lines(img)

        shoreline_result = extract_shoreline(
            img,
            min_cluster_area_m2=args.min_cluster_area,
            morph_open_k=args.morph_open_k,
            morph_close_k=args.morph_close_k,
            grid_size=args.grid_size,
            clockwise_azimuth=AZIMUTH_CLOCKWISE,
            min_segment_points=args.shoreline_min_segment_points,
            min_segment_length_m=args.shoreline_min_segment_length_m,
            return_metadata=args.register_shoreline,
        )
        if args.register_shoreline:
            shoreline_points, cart_mask, polar_mask, cart_mask_float, shoreline_metadata = shoreline_result
        else:
            shoreline_points, cart_mask, polar_mask, cart_mask_float = shoreline_result
            shoreline_metadata = None

        if shoreline_points:
            azimuth_indices, range_indices = zip(*shoreline_points, strict=False)

            n_range_bins = img.shape[1] if img.shape[1] > img.shape[0] else img.shape[0]
            n_azimuth_bins = img.shape[0] if img.shape[1] > img.shape[0] else img.shape[1]

            cart_points = polar_indices_to_cartesian(
                np.array(range_indices),
                np.array(azimuth_indices),
                n_range_bins=n_range_bins,
                n_azimuth_bins=n_azimuth_bins,
                max_range_m=RADAR_MAX_RANGE_M,
                clockwise_azimuth=AZIMUTH_CLOCKWISE,
            )
            cluster_labels = cluster_shoreline_dbscan(cart_points[:, 0], cart_points[:, 1])
        else:
            cart_points = np.empty((0, 2), dtype=float)
            cluster_labels = np.array([])

        image_stem = image_path.stem
        image_plot_name = f"img_{image_index:05d}_{image_stem}_shoreline.png"
        map_plot_name = f"img_{image_index:05d}_{image_stem}_shoreline_osm.png"
        points_name = f"img_{image_index:05d}_{image_stem}_shoreline_points.csv"

        plot_shoreline_extraction(
            img,
            shoreline_points,
            cart_mask,
            polar_mask,
            cluster_labels=cluster_labels,
            output_path=image_dir / image_plot_name,
            clockwise_azimuth=AZIMUTH_CLOCKWISE,
        )

        with (points_dir / points_name).open("w", newline="", encoding="utf-8") as points_file:
            writer = csv.writer(points_file)
            writer.writerow(["azimuth_idx", "range_idx"])
            writer.writerows(shoreline_points)

        pose = None
        corrected_heading_deg = None
        radar_lat = None
        radar_lon = None
        map_plot_relpath = ""
        try:
            image_timestamp = extract_timestamp(image_path.name)
            pose = interpolate_gps_pose(gps_data, image_timestamp)

            if pose is not None:
                corrected_heading_deg = (pose["bearing"] + HEADING_CORRECTION_DEG) % 360.0
                radar_lat, radar_lon = offset_latlon_by_local_offsets(
                    pose["latitude"],
                    pose["longitude"],
                    heading_deg=corrected_heading_deg,
                    forward_m=GPS_TO_RADAR_OFFSET_M,
                    lateral_m=GPS_TO_RADAR_LATERAL_OFFSET_M,
                )

                map_output_path = image_dir / map_plot_name
                plot_radar_overlay_on_osm_static(
                    center_lat=radar_lat,
                    center_lon=radar_lon,
                    heading_deg=corrected_heading_deg,
                    output_path=map_output_path,
                    shoreline_xy=cart_points,
                    cart_mask=cart_mask_float,
                    max_range_m=RADAR_MAX_RANGE_M,
                    title=(
                        f"{image_path.name} | heading={corrected_heading_deg:.1f}° "
                        f"(corr={HEADING_CORRECTION_DEG:+.1f}°)"
                    ),
                )
                map_plot_relpath = f"images/{map_plot_name}"
        except Exception as error:
            print(f"Skipping OSM map for {image_path.name}: {error}")

        registration_plot_relpath = ""
        registration_success = False
        registration_confidence = 0.0
        registration_reason = ""
        registration_dx_m = 0.0
        registration_dy_m = 0.0
        registration_dyaw_deg = 0.0
        registration_inliers = 0
        registration_inlier_ratio = 0.0
        registration_mean_residual_m = np.nan
        registration_observability = ""

        if args.register_shoreline and pose is not None and shoreline_metadata is not None:
            try:
                window_frames = []
                for window_idx in _window_indices(image_index, total_available, args.window_size):
                    window_path = image_files[window_idx]
                    window_img = plt.imread(window_path)
                    if args.correct_black_lines:
                        window_img = correct_black_lines(window_img)

                    _, _, _, _, window_metadata = extract_shoreline(
                        window_img,
                        min_cluster_area_m2=args.min_cluster_area,
                        morph_open_k=args.morph_open_k,
                        morph_close_k=args.morph_close_k,
                        grid_size=args.grid_size,
                        clockwise_azimuth=AZIMUTH_CLOCKWISE,
                        min_segment_points=args.shoreline_min_segment_points,
                        min_segment_length_m=args.shoreline_min_segment_length_m,
                        return_metadata=True,
                    )

                    point_mask = window_metadata["valid_mask"]
                    if not np.any(point_mask):
                        point_mask = window_metadata["quality_weights"] > 0.05
                    if not np.any(point_mask):
                        continue

                    window_timestamp = extract_timestamp(window_path.name)
                    window_pose = interpolate_gps_pose(gps_data, window_timestamp)
                    if window_pose is None:
                        continue

                    window_heading = (window_pose["bearing"] + HEADING_CORRECTION_DEG) % 360.0
                    window_radar_lat, window_radar_lon = offset_latlon_by_local_offsets(
                        window_pose["latitude"],
                        window_pose["longitude"],
                        heading_deg=window_heading,
                        forward_m=GPS_TO_RADAR_OFFSET_M,
                        lateral_m=GPS_TO_RADAR_LATERAL_OFFSET_M,
                    )
                    window_frames.append(
                        {
                            "points": window_metadata["cart_points"][point_mask],
                            "weights": window_metadata["quality_weights"][point_mask],
                            "quality_weights": window_metadata["quality_weights"][point_mask],
                            "segment_ids": window_metadata["segment_ids"][point_mask],
                            "arc_lengths_m": window_metadata["arc_lengths_m"][point_mask],
                            "latitude": window_radar_lat,
                            "longitude": window_radar_lon,
                            "heading_deg": window_heading,
                            "frame_id": window_idx,
                        }
                    )

                if window_frames:
                    anchor_index = min(
                        range(len(window_frames)),
                        key=lambda idx: abs(int(window_frames[idx]["frame_id"]) - int(image_index)),
                    )
                    anchor_frame = window_frames[anchor_index]
                    point_set = accumulate_shoreline_window(
                        window_frames,
                        anchor_index=anchor_index,
                        cell_size_m=args.aggregation_cell_size_m,
                        min_persistence=args.window_min_persistence,
                        min_segment_points=args.shoreline_min_segment_points,
                        min_segment_length_m=args.shoreline_min_segment_length_m,
                    )

                    if point_set.points.shape[0] > 0:
                        cache_path = _coastline_cache_path(
                            args.coastline_cache_dir,
                            anchor_frame["latitude"],
                            anchor_frame["longitude"],
                            args.registration_radius_m,
                        )
                        coastline_cache_key = (
                            round(float(anchor_frame["latitude"]), 5),
                            round(float(anchor_frame["longitude"]), 5),
                            float(args.registration_radius_m),
                            float(args.coastline_sample_step_m),
                            str(args.coastline_geojson or ""),
                        )
                        if coastline_cache_key not in coastline_map_cache:
                            coastline_map_cache[coastline_cache_key] = build_coastline_map(
                                center_latlon=(anchor_frame["latitude"], anchor_frame["longitude"]),
                                radius_m=args.registration_radius_m,
                                sample_step_m=args.coastline_sample_step_m,
                                geojson_path=args.coastline_geojson,
                                cache_path=cache_path,
                            )
                        coastline_map = coastline_map_cache[coastline_cache_key]

                        prior_pose = pose_from_geodetic(
                            anchor_frame["latitude"],
                            anchor_frame["longitude"],
                            anchor_frame["heading_deg"],
                            center_lat=anchor_frame["latitude"],
                            center_lon=anchor_frame["longitude"],
                            transformer_to_local=coastline_map.transformer_to_local,
                        )

                        registration_result = register_shoreline_to_map(
                            point_set,
                            prior_pose,
                            coastline_map,
                            translation_search_m=args.translation_search_m,
                            translation_step_m=args.translation_step_m,
                            rotation_search_deg=args.rotation_search_deg,
                            rotation_step_deg=args.rotation_step_deg,
                        )

                        registration_success = bool(registration_result.success)
                        registration_confidence = float(registration_result.confidence)
                        registration_reason = registration_result.rejection_reason or ""
                        registration_dx_m = float(registration_result.correction.x)
                        registration_dy_m = float(registration_result.correction.y)
                        registration_dyaw_deg = float(np.rad2deg(registration_result.correction.yaw_rad))
                        registration_inliers = int(registration_result.inlier_count)
                        registration_inlier_ratio = float(registration_result.inlier_ratio)
                        registration_mean_residual_m = float(registration_result.mean_abs_residual_m)
                        registration_observability = registration_result.observability

                        registration_plot_name = f"img_{image_index:05d}_{image_stem}_registration.png"
                        plot_shoreline_registration_result(
                            coastline_map=coastline_map,
                            point_set=point_set,
                            prior_pose=prior_pose,
                            registration_result=registration_result,
                            output_path=image_dir / registration_plot_name,
                            title=(
                                f"{image_path.name} | registration success={registration_result.success} "
                                f"| conf={registration_result.confidence:.2f}"
                            ),
                        )
                        registration_plot_relpath = f"images/{registration_plot_name}"
                    else:
                        registration_reason = "no_persistent_points"
                else:
                    registration_reason = "no_window_points"
            except Exception as error:
                registration_reason = str(error)
                print(f"Skipping shoreline registration for {image_path.name}: {error}")

        summary_rows.append(
            {
                "image_index": image_index,
                "image_filename": image_path.name,
                "shoreline_points": len(shoreline_points),
                "image_plot": f"images/{image_plot_name}",
                "points_csv": f"shoreline_points/{points_name}",
                "map_plot": map_plot_relpath,
                "registration_success": registration_success,
                "registration_confidence": registration_confidence,
                "registration_reason": registration_reason,
                "registration_dx_m": registration_dx_m,
                "registration_dy_m": registration_dy_m,
                "registration_dyaw_deg": registration_dyaw_deg,
                "registration_inliers": registration_inliers,
                "registration_inlier_ratio": registration_inlier_ratio,
                "registration_mean_residual_m": registration_mean_residual_m,
                "registration_observability": registration_observability,
                "registration_plot": registration_plot_relpath,
            }
        )

        print(f"Processed {image_index:05d}: {image_path.name} -> {len(shoreline_points)} shoreline points")

    with (run_root / "summary.json").open("w", encoding="utf-8") as summary_json_file:
        json.dump(summary_rows, summary_json_file, indent=2)

    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as summary_csv_file:
        fieldnames = [
            "image_index",
            "image_filename",
            "shoreline_points",
            "image_plot",
            "points_csv",
            "map_plot",
            "registration_success",
            "registration_confidence",
            "registration_reason",
            "registration_dx_m",
            "registration_dy_m",
            "registration_dyaw_deg",
            "registration_inliers",
            "registration_inlier_ratio",
            "registration_mean_residual_m",
            "registration_observability",
            "registration_plot",
        ]
        writer = csv.DictWriter(summary_csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Completed. Wrote {len(summary_rows)} result row(s).")
    print(f"Artifacts saved in: {run_root}")


if __name__ == "__main__":
    main()
