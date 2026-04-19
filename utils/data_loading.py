import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils import DATA_DIR, RESULTS_DIR
else:
    from . import DATA_DIR, RESULTS_DIR

# Folder and file paths to the radar images and GPS data
FOLDER_PATH = DATA_DIR / "_radar_data_b_scan_image"
CSV_FILE_PATH = DATA_DIR / "snowborbus_gps_data.csv"


# Transformation functions for radar data
def polar_to_cartesian_points(
    ranges,
    angles,
    range_resolution=0.155,
    angle_resolution=2 * np.pi / 400,
    clockwise_azimuth=False,
):
    """
    Convert polar coordinates (ranges and angles) to Cartesian coordinates (x, y) and scaling them according to the resolution.
    """
    theta = angles * angle_resolution
    if clockwise_azimuth:
        theta = -theta

    x = ranges * range_resolution * np.cos(theta)
    y = ranges * range_resolution * np.sin(theta)
    points = np.stack((x, y), axis=-1)

    return points


def cartesian_to_polar_points(
    points, range_resolution=0.155, angle_resolution=2 * np.pi / 400, clockwise_azimuth=False
):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (ranges and angles) and scaling them according to the resolution.
    """
    x = points[:, 0]
    y = points[:, 1]
    ranges = np.sqrt(x**2 + y**2) / range_resolution
    theta = np.arctan2(y, x)
    if clockwise_azimuth:
        theta = -theta
    angles = theta / angle_resolution
    polar_points = np.stack((angles, ranges), axis=-1)

    return polar_points


def _normalize_polar_layout(data, data_layout="auto"):
    """Return data in (range, azimuth) layout."""
    if data.ndim != 2:
        raise ValueError("Expected a 2D polar radar image")

    if data_layout not in {"auto", "range_azimuth", "azimuth_range"}:
        raise ValueError("data_layout must be 'auto', 'range_azimuth', or 'azimuth_range'")

    if data_layout == "range_azimuth":
        return data
    if data_layout == "azimuth_range":
        return data.T

    # Auto: assume range axis is the longer dimension for these PNG captures
    return data.T if data.shape[1] > data.shape[0] else data


def polar_to_cartesian_image(
    data,
    theta_range=None,
    r_range=(0, 1000),
    grid_size=1024,
    data_layout="auto",
    clockwise_azimuth=False,
):
    """
    Convert radar image from polar to Cartesian coordinates.

    Parameters:
    -----------
    data : ndarray
        2D array in polar format
    theta_range : tuple, optional
        (min_angle, max_angle) in radians
    r_range : tuple, optional
        (min_range, max_range) in meters/samples
    grid_size : int, optional
        Size of output Cartesian grid
    data_layout : str, optional
        'range_azimuth', 'azimuth_range', or 'auto'
    """
    polar_data = _normalize_polar_layout(data, data_layout=data_layout)
    n_range, n_azimuth = polar_data.shape

    if theta_range is None:
        theta_range = (0, 2 * np.pi)
    if r_range is None:
        r_range = (0, n_range - 1)

    r_min, r_max = r_range
    theta_min, theta_max = theta_range

    x = np.linspace(-r_max, r_max, grid_size, dtype=np.float32)
    y = np.linspace(-r_max, r_max, grid_size, dtype=np.float32)
    X_cart, Y_cart = np.meshgrid(x, y)

    r = np.sqrt(X_cart**2 + Y_cart**2)
    theta = np.arctan2(Y_cart, X_cart)
    if clockwise_azimuth:
        theta = -theta
    theta = np.mod(theta - theta_min, 2 * np.pi) + theta_min

    in_range = (r >= r_min) & (r <= r_max)
    if theta_max > theta_min:
        in_theta = (theta >= theta_min) & (theta <= theta_max)
    else:
        # wrapped interval, e.g. (3π/2, π/2)
        in_theta = (theta >= theta_min) | (theta <= theta_max)
    valid = in_range & in_theta

    r_idx = (r - r_min) * (n_range - 1) / (r_max - r_min + 1e-12)
    theta_span = (theta_max - theta_min) if theta_max > theta_min else (theta_max + 2 * np.pi - theta_min)
    theta_unwrapped = np.mod(theta - theta_min, 2 * np.pi)
    theta_idx = theta_unwrapped * (n_azimuth - 1) / (theta_span + 1e-12)
    theta_idx = np.mod(theta_idx, n_azimuth)

    sampled = map_coordinates(
        polar_data,
        [r_idx.ravel(), theta_idx.ravel()],
        order=1,
        mode="wrap",
        prefilter=False,
    ).reshape(grid_size, grid_size)

    sampled[~valid] = 0.0

    return sampled.astype(np.float32), x, y


def cartesian_to_polar_image(cart_data, polar_shape, theta_range=None, r_range=None, clockwise_azimuth=False):
    """
    Convert a radar image from Cartesian to polar coordinates.

    Parameters:
    -----------
    cart_data : ndarray
        2D array in Cartesian format (grid_size x grid_size)
    polar_shape : tuple
        (n_range, n_azimuth) dimensions of the target polar image
    theta_range : tuple, optional
        (min_angle, max_angle) in radians
    r_range : tuple, optional
        (min_range, max_range) in meters/samples

    Returns:
    --------
    polar_data : ndarray
        2D array in polar format (n_range x n_azimuth)
    """
    n_range, n_azimuth = polar_shape
    grid_size = cart_data.shape[0]

    if theta_range is None:
        theta_range = (0, 2 * np.pi)
    if r_range is None:
        r_range = (0, n_range - 1)

    r_min, r_max = r_range
    theta_min, theta_max = theta_range

    r_polar = np.linspace(r_min, r_max, n_range, dtype=np.float32)

    # Calculate azimuth angles
    theta_span = (theta_max - theta_min) if theta_max > theta_min else (theta_max + 2 * np.pi - theta_min)
    theta_polar = np.linspace(theta_min, theta_min + theta_span, n_azimuth, endpoint=False, dtype=np.float32)

    R, Theta = np.meshgrid(r_polar, theta_polar, indexing="ij")

    # Convert to Cartesian coords (-r_max to r_max is mapped to 0 to grid_size)
    theta_math = -Theta if clockwise_azimuth else Theta
    X = R * np.cos(theta_math)
    Y = R * np.sin(theta_math)

    # Calculate indices in the Cartesian grid
    x_idx = (X - (-r_max)) * (grid_size - 1) / (r_max - (-r_max) + 1e-12)
    y_idx = (Y - (-r_max)) * (grid_size - 1) / (r_max - (-r_max) + 1e-12)

    sampled = map_coordinates(
        cart_data,
        [y_idx.ravel(), x_idx.ravel()],  # order is Y (row), X (col)
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(n_range, n_azimuth)

    return sampled.astype(cart_data.dtype)


# Function to extract timestamp from filename
def extract_timestamp(filename):
    """Extract timestamp from filename.

    Supported patterns include:
    - radar_YYYY-MM-DD_HH-MM-SS_ms.png
    - img_<idx>_frame_YYYY-MM-DD_HH-MM-SS_ms*.png
    """
    name = Path(filename).name
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_(\d{1,6})", name)
    if not match:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")

    date_part, hour, minute, second, fractional = match.groups()
    # Normalize to milliseconds to keep compatibility with existing GPS timestamp precision
    fractional_ms = fractional[:3].ljust(3, "0")
    return f"{date_part}T{hour}:{minute}:{second}.{fractional_ms}"


def extract_gps_timeframe(gps_data, timestamps, time_tolerance=0.5):
    """Extract GPS rows within [timestamp - tol, timestamp] for each radar timestamp."""
    gps_subset = pd.DataFrame()

    for timestamp in timestamps:
        radar_time_point = pd.to_datetime(timestamp)
        time_window_start = radar_time_point - pd.Timedelta(seconds=time_tolerance)
        time_window_end = radar_time_point

        subset = gps_data[(gps_data["time-string"] >= time_window_start) & (gps_data["time-string"] <= time_window_end)]
        if subset.empty:
            continue
        gps_subset = pd.concat([gps_subset, subset], ignore_index=True)

    return gps_subset.drop_duplicates().reset_index(drop=True)


def _interpolate_bearing_deg(bearing_before_deg, bearing_after_deg, ratio):
    """Interpolate compass bearings along shortest angular path."""
    if pd.isna(bearing_before_deg) and pd.isna(bearing_after_deg):
        return np.nan
    if pd.isna(bearing_before_deg):
        return float(bearing_after_deg)
    if pd.isna(bearing_after_deg):
        return float(bearing_before_deg)

    before = float(bearing_before_deg) % 360.0
    after = float(bearing_after_deg) % 360.0
    delta = ((after - before + 180.0) % 360.0) - 180.0
    return (before + ratio * delta) % 360.0


def calculate_bearing_from_latlon(lat1, lon1, lat2, lon2):
    """Calculate compass bearing (degrees) from (lat1, lon1) to (lat2, lon2)."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    diff_long_rad = np.radians(lon2 - lon1)

    x = np.sin(diff_long_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(diff_long_rad)

    initial_bearing = np.degrees(np.arctan2(x, y))
    return (initial_bearing + 360.0) % 360.0


def interpolate_gps_pose(gps_data, timestamp):
    """Interpolate GPS pose (lat, lon, bearing) for a single timestamp.

    Returns a dict with keys: 'time-string', 'latitude', 'longitude', 'bearing'.
    """
    gps_sorted = (
        gps_data.dropna(subset=["time-string", "latitude", "longitude"])
        .sort_values("time-string")
        .reset_index(drop=True)
    )
    if gps_sorted.empty:
        return None

    radar_time_point = pd.to_datetime(timestamp)
    before_candidates = gps_sorted[gps_sorted["time-string"] <= radar_time_point]
    after_candidates = gps_sorted[gps_sorted["time-string"] >= radar_time_point]

    if before_candidates.empty and after_candidates.empty:
        return None

    if before_candidates.empty:
        nearest = after_candidates.iloc[0]
        return {
            "time-string": radar_time_point,
            "latitude": float(nearest["latitude"]),
            "longitude": float(nearest["longitude"]),
            "bearing": float(nearest["bearing"])
            if "bearing" in gps_sorted.columns and not pd.isna(nearest.get("bearing"))
            else np.nan,
        }

    if after_candidates.empty:
        nearest = before_candidates.iloc[-1]
        return {
            "time-string": radar_time_point,
            "latitude": float(nearest["latitude"]),
            "longitude": float(nearest["longitude"]),
            "bearing": float(nearest["bearing"])
            if "bearing" in gps_sorted.columns and not pd.isna(nearest.get("bearing"))
            else np.nan,
        }

    before = before_candidates.iloc[-1]
    after = after_candidates.iloc[0]

    if before["time-string"] == after["time-string"]:
        lat_interp = float(before["latitude"])
        lon_interp = float(before["longitude"])
        bearing_interp = (
            float(before["bearing"])
            if "bearing" in gps_sorted.columns and not pd.isna(before.get("bearing"))
            else np.nan
        )
    else:
        total_time_diff = (after["time-string"] - before["time-string"]).total_seconds()
        if total_time_diff <= 0:
            lat_interp = float(before["latitude"])
            lon_interp = float(before["longitude"])
            bearing_interp = (
                float(before["bearing"])
                if "bearing" in gps_sorted.columns and not pd.isna(before.get("bearing"))
                else np.nan
            )
        else:
            ratio = (radar_time_point - before["time-string"]).total_seconds() / total_time_diff
            lat_interp = float(before["latitude"] + ratio * (after["latitude"] - before["latitude"]))
            lon_interp = float(before["longitude"] + ratio * (after["longitude"] - before["longitude"]))
            bearing_interp = (
                _interpolate_bearing_deg(before.get("bearing", np.nan), after.get("bearing", np.nan), ratio)
                if "bearing" in gps_sorted.columns
                else np.nan
            )

    if pd.isna(bearing_interp):
        bearing_interp = calculate_bearing_from_latlon(
            float(before["latitude"]), float(before["longitude"]), float(after["latitude"]), float(after["longitude"])
        )

    return {
        "time-string": radar_time_point,
        "latitude": lat_interp,
        "longitude": lon_interp,
        "bearing": float(bearing_interp),
    }


def offset_latlon_by_distance_bearing(latitude, longitude, distance_m, bearing_deg):
    """Move a geographic point by distance (m) along compass bearing (deg)."""
    earth_radius_m = 6378137.0
    angular_distance = float(distance_m) / earth_radius_m
    bearing_rad = np.radians(float(bearing_deg))

    lat1 = np.radians(float(latitude))
    lon1 = np.radians(float(longitude))

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance) + np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing_rad)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat1),
        np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2),
    )

    lon2 = (lon2 + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(lat2)), float(np.degrees(lon2))


def offset_latlon_by_local_offsets(latitude, longitude, heading_deg, forward_m=0.0, lateral_m=0.0):
    """Offset a GPS coordinate by local vehicle-frame offsets.

    Parameters:
    - heading_deg: compass heading (0° north, 90° east)
    - forward_m: positive in heading direction
    - lateral_m: positive to the right of heading
    """
    lat_out, lon_out = float(latitude), float(longitude)

    if abs(float(forward_m)) > 0.0:
        lat_out, lon_out = offset_latlon_by_distance_bearing(lat_out, lon_out, float(forward_m), float(heading_deg))

    if abs(float(lateral_m)) > 0.0:
        lat_out, lon_out = offset_latlon_by_distance_bearing(
            lat_out,
            lon_out,
            abs(float(lateral_m)),
            (float(heading_deg) + (90.0 if lateral_m >= 0 else -90.0)) % 360.0,
        )

    return lat_out, lon_out


def polar_indices_to_cartesian(
    ranges_idx,
    azimuths_idx,
    n_range_bins,
    n_azimuth_bins,
    max_range_m=1000.0,
    clockwise_azimuth=False,
):
    """Convert polar index arrays to local Cartesian meters using radar geometry."""
    ranges_idx = np.asarray(ranges_idx, dtype=float)
    azimuths_idx = np.asarray(azimuths_idx, dtype=float)

    if int(n_range_bins) <= 1 or int(n_azimuth_bins) <= 0:
        raise ValueError("n_range_bins must be > 1 and n_azimuth_bins must be > 0")

    range_resolution_m = float(max_range_m) / float(int(n_range_bins) - 1)
    angle_resolution_rad = (2.0 * np.pi) / float(int(n_azimuth_bins))

    return polar_to_cartesian_points(
        ranges_idx,
        azimuths_idx,
        range_resolution=range_resolution_m,
        angle_resolution=angle_resolution_rad,
        clockwise_azimuth=clockwise_azimuth,
    )


# Load functions for radar images and GPS data
def load_radar_images(num_images=5, folder_path=FOLDER_PATH, correct_black_lines_flag=True):
    # List all PNG files in the folder and sort them by timestamp
    image_files = sorted([f.name for f in Path(folder_path).iterdir() if f.suffix == ".png"], key=extract_timestamp)

    # Load the first num_images radar images
    images = []
    for i in range(min(num_images, len(image_files))):
        img = plt.imread(Path(folder_path) / image_files[i])

        if correct_black_lines_flag:
            img = correct_black_lines(img)

        images.append(img)

    return image_files, images


def load_gps_data(csv_file_path=CSV_FILE_PATH):
    gps_data = pd.read_csv(csv_file_path)
    gps_data["time-string"] = pd.to_datetime(gps_data["time-string"])
    return gps_data


# Add function which corrects the blank space in radar images
def correct_black_lines(image, min_black_line_width=50):
    """
    Corrects fragmented data with black horizontal lines (consecutive missing data) in the middle of images.
    If any data to the right of the black line is present, it shifts the data to the left, essentially moving the black line to the right.
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray_image = image.copy()

    # Identify black lines
    black_spots_mask = gray_image == 0

    # find connected black line segments and their widths
    for row in range(black_spots_mask.shape[0]):
        black_line_segments = np.split(black_spots_mask[row], np.where(np.diff(black_spots_mask[row]))[0] + 1)
        current_col = gray_image.shape[1] - 1  # Start from the rightmost column
        last_valid_col = gray_image.shape[1] - 1
        for segment in reversed(black_line_segments):  # Process from right to left
            if len(segment) > 0 and segment[0]:  # If the segment starts with a black line
                line_width = len(segment)
                if line_width >= min_black_line_width:
                    # Shift the data to the left by the width of existing data to the right of the black line
                    data_to_shift_width = last_valid_col - current_col
                    if data_to_shift_width > 0:
                        gray_image[
                            row, current_col - line_width + 1 : current_col - line_width + 1 + data_to_shift_width
                        ] = gray_image[row, current_col + 1 : current_col + 1 + data_to_shift_width]
                        # Fill the vacated area with zeros (black)
                        gray_image[
                            row,
                            current_col + 1 + data_to_shift_width - line_width : current_col + 1 + data_to_shift_width,
                        ] = 0
                        last_valid_col -= line_width  # Update previous end column after shifting

            current_col -= len(segment)

    return gray_image


if __name__ == "__main__":
    # Load radar image
    image_file, image = load_radar_images(num_images=4, correct_black_lines_flag=True)
    img = image[3]

    time_stamp = extract_timestamp(image_file[3])
    print(f"Loaded radar image with timestamp: {time_stamp}")

    cartesian_image, x, y = polar_to_cartesian_image(img)

    plt.figure(figsize=(12, 12))

    plt.subplot(1, 2, 1)
    plt.title("Radar Image (Polar)")
    plt.imshow(img, cmap="gray", aspect="auto")

    plt.subplot(1, 2, 2)
    plt.title("Radar Image (Cartesian)")
    plt.imshow(cartesian_image, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap="gray")

    # if graphical backend is not available, save the figure instead of showing it
    import matplotlib

    if matplotlib.is_interactive() or matplotlib.get_backend().lower() not in {"agg", "svg", "pdf", "ps", "cairo"}:
        plt.show()
    else:
        out = RESULTS_DIR / "radar_image_visualization.png"
        plt.savefig(out)
        print(f"Graphical backend not available, saved figure to {out}")
