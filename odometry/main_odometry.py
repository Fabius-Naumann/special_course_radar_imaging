import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from odometry.ablation_study import main_odometry
from odometry.evaluation import map_gps
from utils.data_loading import load_gps_data

def main():
    print("Starting odometry evaluation pipeline...")
    # Default parameters based on notebook/ablation_study.py
    params = {
        "n_images": 251,
        "preprocessing": "normalized_azimuths",
        "k": 20,
        "max_distance_percentile": 25,
        "r_param": 3.0,
        "f_param": 1.0,
        "z_percentile": 99.0,
        "cost_function": "p2p",
        "window_size": 3,
        "motion_compensation_flag": True,
        "iterations": 8,
        "every_nth_frame": 1,
        "print_lines": True,
        "plot_map": False,
        "smoothing": None,
        "clustering": None,
        "dataset": 1,
        "stop_if_pos_fault_gt_m": 50.0
    }

    print(f"Running odometry with parameters: {params}")
    odometry_results, timestamps, used_images = main_odometry(**params)

    # Save results to CSV
    output_dir = PROJECT_ROOT / "outputs" / "odometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "odometry_results.csv"
    odometry_results.to_csv(csv_path, index=False)
    print(f"Saved odometry results to {csv_path}")

    # Plot GPS Map
    print("Generating GPS map...")
    gps_data = load_gps_data()
    map_output_file = str(output_dir / "gps_map.html")
    
    odo_lat = odometry_results["odometry_latitude"].values
    odo_lon = odometry_results["odometry_longitude"].values
    
    # We pass the absolute longitudes and latitudes for odometry
    map_gps(
        gps_data=gps_data,
        odo_trans=np.array([[0,0]]),
        timestamps=timestamps,
        odo_longitudes=odo_lon,
        odo_latitudes=odo_lat,
        apply_rigid_body_correction=False,
        output_file=map_output_file
    )

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
