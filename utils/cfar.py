import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import rank_filter

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loading import load_radar_images, polar_to_cartesian_image
from utils.visualisation import plot_radars_side_by_side

# =============================================================================
# Polar CFAR utilities for radar intensity images
# -----------------------------------------------------------------------------
# Assumptions:
# - input image shape: (n_azimuth, n_range)
# - axis 0 = azimuth
# - axis 1 = range
# - values are nonnegative intensity-like radar values
# - azimuth wraps around cyclically
# - range does NOT wrap
#
# Notes:
# - CA/GOCA/SOCA use summed-area / integral-image style implementations and are
#   memory-safe for images like 400 x 6000.
# - OS-CFAR is implemented with scipy.ndimage.rank_filter. It is much heavier
#   than CA/GOCA/SOCA but avoids the catastrophic memory blow-up from explicit
#   sliding windows.
# - If your data is not true calibrated power, the "pfa" parameter is only an
#   approximate tuning knob, not a guaranteed false alarm probability.
# =============================================================================

RANGE_BIN_M = 0.155  # meters per range bin, adjust as needed for your radar
CFAR_PFA = 2e-1

# =============================================================================
# Helper functions
# =============================================================================


def _alpha_ca_from_pfa(n_train, pfa):
    """
    Classical CA-CFAR threshold factor for exponential / square-law data.

    For uncalibrated intensity data this is only approximate, but still a
    useful starting point.
    """
    n_train = np.asarray(n_train, dtype=np.float64)
    alpha = np.full_like(n_train, np.nan, dtype=np.float64)
    valid = n_train > 0
    alpha[valid] = n_train[valid] * (pfa ** (-1.0 / n_train[valid]) - 1.0)
    return alpha


def _pad_polar_image(img, pad_az, pad_rg, range_pad_mode="edge"):
    """
    Pad polar image:
    - azimuth wraps cyclically
    - range is padded non-cyclically
    """
    img = np.asarray(img)
    padded = np.pad(img, ((pad_az, pad_az), (0, 0)), mode="wrap")
    padded = np.pad(padded, ((0, 0), (pad_rg, pad_rg)), mode=range_pad_mode)
    return padded


def _integral_image(img):
    """
    OpenCV integral image. Output shape is (H+1, W+1).
    """
    return cv2.integral(img.astype(np.float64))


def _box_sum_from_integral(ii, top, left, bottom, right):
    """
    Vectorized rectangle sums using an integral image.

    Rectangle convention:
    [top:bottom, left:right], with bottom/right exclusive.
    """
    return ii[bottom, right] - ii[top, right] - ii[bottom, left] + ii[top, left]


def _center_indices_for_original(img_shape, pad_az, pad_rg):
    """
    For original image shape (H, W), return center indices inside padded image.
    """
    h, w = img_shape
    rows, cols = np.mgrid[0:h, 0:w]
    center_r = rows + pad_az
    center_c = cols + pad_rg
    return center_r, center_c


def _crop_valid_from_padded(shape, pad_az, pad_rg, arr):
    """
    Crop padded array back to original image support.
    """
    h, w = shape
    return arr[pad_az : pad_az + h, pad_rg : pad_rg + w]


def _check_input_image(img):
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    if np.any(img < 0):
        raise ValueError("CFAR input should be nonnegative.")
    return img.astype(np.float32, copy=False)


def _normalize_azimuth_rows(img):
    """Normalize each azimuth row by its mean or median to mitigate large-scale
    variations in antenna pattern or scene structure. This is optional and not part of the core CFAR logic."""
    row_medians = np.median(img, axis=1, keepdims=True)
    # Compute median over nonzero values in each row
    row_medians = np.array([np.median(row[row != 0]) for row in img]).reshape(-1, 1)
    row_medians[row_medians == 0] = 1.0  # Avoid division by zero
    normalized = img / row_medians
    return normalized.astype(np.float32, copy=False)


# =============================================================================
# CA-CFAR
# =============================================================================


def cfar2d_polar_ca(
    img,
    guard_az,
    guard_rg,
    train_az,
    train_rg,
    pfa=1e-3,
    range_pad_mode="edge",
    normalize_azimuth=False,
    min_noise_floor_factor=0,
):
    """
    Efficient 2D CA-CFAR for polar radar intensity images.

    Parameters
    ----------
    img : ndarray, shape (n_az, n_rg)
        Nonnegative polar intensity image.
    guard_az, guard_rg : int
        Guard half-width in azimuth and range bins.
    train_az, train_rg : int
        Training-band thickness beyond the guard region.
    pfa : float
        Desired false alarm probability. Approximate for non-power data.
    range_pad_mode : str
        Padding mode for range axis, e.g. "edge", "reflect", "constant".
    normalize_azimuth : bool
        If True, normalize each azimuth row by its median to mitigate large-scale
        variations in antenna pattern or scene structure.
    min_noise_floor_factor : float, optional
        Factor to multiply with the median of the image to set the minimum noise floor.
    Returns
    -------
    detections : bool ndarray
    threshold  : float32 ndarray
    noise      : float32 ndarray
    valid      : bool ndarray
    """
    img = _check_input_image(img)

    if guard_az < 0 or guard_rg < 0 or train_az < 0 or train_rg < 0:
        raise ValueError("guard_* and train_* must be nonnegative.")
    if train_az == 0 and train_rg == 0:
        raise ValueError("At least one training extent must be > 0.")

    img = _normalize_azimuth_rows(img) if normalize_azimuth else img

    pad_az = guard_az + train_az
    pad_rg = guard_rg + train_rg

    padded = _pad_polar_image(img, pad_az, pad_rg, range_pad_mode=range_pad_mode)
    ii = _integral_image(padded)

    h, w = img.shape
    cr, cc = _center_indices_for_original((h, w), pad_az, pad_rg)

    # Outer rectangle
    ot = cr - pad_az
    ol = cc - pad_rg
    ob = cr + pad_az + 1
    or_ = cc + pad_rg + 1

    # Inner rectangle = guard region + CUT
    it = cr - guard_az
    il = cc - guard_rg
    ib = cr + guard_az + 1
    ir = cc + guard_rg + 1

    outer_sum = _box_sum_from_integral(ii, ot, ol, ob, or_)
    inner_sum = _box_sum_from_integral(ii, it, il, ib, ir)

    outer_cnt = (2 * pad_az + 1) * (2 * pad_rg + 1)
    inner_cnt = (2 * guard_az + 1) * (2 * guard_rg + 1)
    n_train = outer_cnt - inner_cnt

    if n_train <= 0:
        raise ValueError("No training cells left. Increase training or reduce guard.")

    noise = (outer_sum - inner_sum) / n_train
    alpha = _alpha_ca_from_pfa(np.full((h, w), n_train, dtype=np.float64), pfa)
    threshold = alpha * noise
    threshold = np.maximum(threshold, min_noise_floor_factor * np.median(img))  # ensure minimal noise-floor is removed
    detections = img > threshold
    valid = np.ones_like(detections, dtype=bool)

    return detections, threshold.astype(np.float32), noise.astype(np.float32), valid


# =============================================================================
# OS-CFAR
# =============================================================================


def cfar2d_polar_os(
    img,
    guard_az,
    guard_rg,
    train_az,
    train_rg,
    rank_k=None,
    alpha=1.5,
    range_pad_mode="edge",
    normalize_azimuth=False,
    min_noise_floor_factor=0,
):
    """
    Memory-safe 2D OS-CFAR for polar radar intensity images.

    Parameters
    ----------
    img : ndarray, shape (n_az, n_rg)
    guard_az, guard_rg : int
    train_az, train_rg : int
    rank_k : int or None
        1-based rank among training cells.
        If None, uses approximately the 75th percentile.
    alpha : float
        Multiplier applied to the selected order statistic.
        Must be tuned empirically for intensity-like data.

    Returns
    -------
    detections, threshold, noise, valid
    """
    img = _check_input_image(img)

    if guard_az < 0 or guard_rg < 0 or train_az < 0 or train_rg < 0:
        raise ValueError("guard_* and train_* must be nonnegative.")
    if train_az == 0 and train_rg == 0:
        raise ValueError("At least one training extent must be > 0.")

    img = _normalize_azimuth_rows(img) if normalize_azimuth else img

    pad_az = guard_az + train_az
    pad_rg = guard_rg + train_rg

    padded = _pad_polar_image(img, pad_az, pad_rg, range_pad_mode=range_pad_mode)

    win_h = 2 * pad_az + 1
    win_w = 2 * pad_rg + 1

    footprint = np.ones((win_h, win_w), dtype=bool)

    # Remove guard region + CUT
    az0 = pad_az - guard_az
    az1 = pad_az + guard_az + 1
    rg0 = pad_rg - guard_rg
    rg1 = pad_rg + guard_rg + 1
    footprint[az0:az1, rg0:rg1] = False

    n_train = int(footprint.sum())
    if n_train <= 0:
        raise ValueError("No training cells left. Increase training or reduce guard.")

    if rank_k is None:
        rank_k = int(np.ceil(0.75 * n_train))

    if not (1 <= rank_k <= n_train):
        raise ValueError(f"rank_k must be in [1, {n_train}]")

    # scipy rank_filter uses 0-based rank
    rank0 = rank_k - 1

    ordered_stat_padded = rank_filter(
        padded,
        rank=rank0,
        footprint=footprint,
        mode="nearest",  # padding already handled explicitly
    )

    noise = _crop_valid_from_padded(img.shape, pad_az, pad_rg, ordered_stat_padded)
    threshold = alpha * noise
    threshold = np.maximum(threshold, min_noise_floor_factor * np.median(img))  # ensure minimal noise-floor is removed
    detections = img > threshold
    valid = np.ones_like(detections, dtype=bool)

    return detections, threshold.astype(np.float32), noise.astype(np.float32), valid


# =============================================================================
# Optional helpers for parameter conversion
# =============================================================================


def range_m_to_bins(meters, range_bin_m):
    """
    Convert a physical range extent in meters to range bins.
    """
    return max(0, round(meters / range_bin_m))


def suggest_default_params(range_bin_m=RANGE_BIN_M):
    """
    Conservative starting point for a radar like yours.
    """
    params = {
        "guard_az": 2,
        "train_az": 10,
        "guard_rg": max(1, round(3.0 / range_bin_m)),  # ~1.5 m
        "train_rg": max(2, round(15.0 / range_bin_m)),  # ~6 m
    }
    return params


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Synthetic example only. Replace this with your real polar image:
    # shape should be (n_azimuth, n_range), e.g. (400, 6000)

    filenames, images = load_radar_images(num_images=1)
    polar_img = images[0]
    cartesian_data, x, y = polar_to_cartesian_image(polar_img)
    n_az = polar_img.shape[0]

    params = suggest_default_params(range_bin_m=RANGE_BIN_M)
    print("Suggested params:", params)

    det_ca, thr_ca, noise_ca, valid_ca = cfar2d_polar_ca(
        polar_img,
        **(params | {"normalize_azimuth": True}),
        range_pad_mode="edge",
        pfa=CFAR_PFA,
        min_noise_floor_factor=0.8,
    )
    print(f"CA detections: {det_ca.sum()}")

    det_os, thr_os, noise_os, valid_os = cfar2d_polar_os(
        polar_img,
        **(params | {"normalize_azimuth": True}),
        range_pad_mode="edge",
        rank_k=None,
        alpha=1.3,
        min_noise_floor_factor=0.8,
    )
    print(f"OS detections: {det_os.sum()}")

    plot_radars_side_by_side(
        [_normalize_azimuth_rows(polar_img), thr_ca, det_ca.astype(float), thr_os, det_os.astype(float)],
        titles=[
            "Polar Image",
            "CA-CFAR Threshold",
            "CA-CFAR Detections",
            "OS-CFAR Threshold",
            "OS-CFAR Detections",
        ],
        fig_size=5,
        filename="cfar_cartesian_comparison.png",
    )
