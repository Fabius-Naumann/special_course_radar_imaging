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

from map.shore_detect import cluster_shoreline_dbscan, extract_shoreline
from utils import RESULTS_DIR
from utils.data_loading import FOLDER_PATH, correct_black_lines, polar_to_cartesian_points
from utils.visualisation import plot_shoreline_extraction


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
        "correct_black_lines": args.correct_black_lines,
        "total_available_images": total_available,
        "selected_indices": selected_indices,
    }

    with open(run_root / "run_config.json", "w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, indent=2)

    print(f"Run directory: {run_root}")
    print(f"Selected {len(selected_indices)} image(s) out of {total_available} available.")

    summary_rows = []
    for image_index in selected_indices:
        image_path = image_files[image_index]
        img = plt.imread(image_path)
        if args.correct_black_lines:
            img = correct_black_lines(img)

        shoreline_points, cart_mask, polar_mask = extract_shoreline(
            img,
            min_cluster_area_m2=args.min_cluster_area,
            morph_open_k=args.morph_open_k,
            morph_close_k=args.morph_close_k,
            grid_size=args.grid_size,
        )

        if shoreline_points:
            azimuth_indices, range_indices = zip(*shoreline_points)
            cart_points = polar_to_cartesian_points(np.array(range_indices), np.array(azimuth_indices))
            cluster_labels = cluster_shoreline_dbscan(cart_points[:, 0], cart_points[:, 1])
        else:
            cluster_labels = np.array([])

        image_stem = image_path.stem
        image_plot_name = f"img_{image_index:05d}_{image_stem}_shoreline.png"
        points_name = f"img_{image_index:05d}_{image_stem}_shoreline_points.csv"

        plot_shoreline_extraction(
            img,
            shoreline_points,
            cart_mask,
            polar_mask,
            cluster_labels=cluster_labels,
            output_path=image_dir / image_plot_name,
        )

        with open(points_dir / points_name, "w", newline="", encoding="utf-8") as points_file:
            writer = csv.writer(points_file)
            writer.writerow(["azimuth_idx", "range_idx"])
            writer.writerows(shoreline_points)

        summary_rows.append(
            {
                "image_index": image_index,
                "image_filename": image_path.name,
                "shoreline_points": len(shoreline_points),
                "image_plot": f"images/{image_plot_name}",
                "points_csv": f"shoreline_points/{points_name}",
            }
        )

        print(f"Processed {image_index:05d}: {image_path.name} -> {len(shoreline_points)} shoreline points")

    with open(run_root / "summary.json", "w", encoding="utf-8") as summary_json_file:
        json.dump(summary_rows, summary_json_file, indent=2)

    with open(run_root / "summary.csv", "w", newline="", encoding="utf-8") as summary_csv_file:
        fieldnames = ["image_index", "image_filename", "shoreline_points", "image_plot", "points_csv"]
        writer = csv.DictWriter(summary_csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Completed. Wrote {len(summary_rows)} result row(s).")
    print(f"Artifacts saved in: {run_root}")


if __name__ == "__main__":
    main()
