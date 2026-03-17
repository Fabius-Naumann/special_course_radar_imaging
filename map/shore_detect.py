import data_loading
import matplotlib.pyplot as plt
import numpy as np
from cfar import CFAR_PFA, cfar2d_polar_ca, suggest_default_params
from data_loading import cartesian_to_polar_image, polar_to_cartesian_image
from scipy.ndimage import binary_closing, binary_opening, label


def extract_shoreline(img, min_cluster_area=4, morph_open_k=3, morph_close_k=5, grid_size=4096):
    """
    Extract shoreline points from the radar image.

    Parameters:
    - img: 2D array of the radar image in polar coordinates.
    - min_cluster_area: Minimum area (in pixels) for a cluster of detections to be kept.
    - morph_open_k: Structure size for morphological opening (removes isolated noise).
    - morph_close_k: Structure size for morphological closing (fills gaps).

    Returns:
    - shoreline_points: List of tuples (azimuth_idx, range_idx) for detected shoreline points.
    """
    # 1. CFAR Detection
    detections, threshold, noise, valid = cfar2d_polar_ca(
        img,
        **suggest_default_params(),
        normalize_azimuth=True,
        range_pad_mode="edge",
        pfa=CFAR_PFA,
        min_noise_floor_factor=1.3,
    )

    # 2. Polar to Cartesian mask conversion
    # polar_to_cartesian_image manages the transpose automatically via _normalize_polar_layout
    cart_mask_float, x, y = polar_to_cartesian_image(detections.astype(np.float32), grid_size=grid_size)
    cart_mask = cart_mask_float > 0.0

    # 3. Morphological filtering in Cartesian
    if morph_close_k > 0:
        struct_close = np.ones((morph_close_k, morph_close_k), dtype=bool)
        cart_mask = binary_closing(cart_mask, structure=struct_close, iterations=1)

    if morph_open_k > 0:
        struct_open = np.ones((morph_open_k, morph_open_k), dtype=bool)
        cart_mask = binary_opening(cart_mask, structure=struct_open, iterations=1)

    # Area filtering
    if min_cluster_area > 0:
        labeled_mask, n_features = label(cart_mask)
        clean_mask = np.zeros_like(cart_mask)
        for i in range(1, n_features + 1):
            if (labeled_mask == i).sum() >= min_cluster_area:
                clean_mask[labeled_mask == i] = True
        cart_mask = clean_mask

    # 4. Cartesian to Polar mapping
    n_range, n_azimuth = data_loading._normalize_polar_layout(img).shape
    polar_mask_normalized = cartesian_to_polar_image(cart_mask.astype(np.float32), polar_shape=(n_range, n_azimuth))
    polar_mask_normalized = polar_mask_normalized > 0.5

    # Match original shape
    if img.shape[0] == n_azimuth:  # meaning original was (azimuth, range)
        polar_mask = polar_mask_normalized.T
        az_axis = 0
        r_axis = 1
    else:
        polar_mask = polar_mask_normalized
        az_axis = 1
        r_axis = 0

    # 5. Extract shoreline points (first true detection in each azimuth)
    shoreline_points = []

    for az in range(n_azimuth):
        # Slice to get all range bins for this azimuth
        range_slice = polar_mask[az, :] if az_axis == 0 else polar_mask[:, az]

        range_indices = np.where(range_slice)[0]
        if len(range_indices) > 0:
            first_range = range_indices[0]  # lowest range index
            # Append (az_idx, range_idx)
            shoreline_points.append((az, first_range))

    return shoreline_points, cart_mask, polar_mask


if __name__ == "__main__":
    from data_loading import load_radar_images, polar_to_cartesian_points
    from visualisation import _display_or_save

    print("Loading radar images...")
    filenames, images = load_radar_images(num_images=1)
    img = images[0]

    print("Extracting shoreline...")
    shoreline_points, cart_mask, polar_mask = extract_shoreline(img)

    print(f"Detected {len(shoreline_points)} shoreline points.")

    # Convert the full image to Cartesian for the background
    cart_img, x_coords, y_coords = polar_to_cartesian_image(img, grid_size=1024)

    # Transform the detected shoreline polar coordinates to Cartesian directly
    cart_shoreline_x = []
    cart_shoreline_y = []
    if shoreline_points:
        azs, ranges = zip(*shoreline_points)
        # polar_to_cartesian_points takes (ranges, angles)
        pts = polar_to_cartesian_points(np.array(ranges), np.array(azs))
        cart_shoreline_x = pts[:, 0]
        cart_shoreline_y = pts[:, 1]

    # Plotting the results using matplotlib to allow custom overlays
    # while leveraging the _display_or_save util for backend handling
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # 1. Original Polar Image with Shoreline
    axs[0].imshow(img.T if img.shape[1] > img.shape[0] else img, aspect="auto", cmap="viridis", origin="lower")
    axs[0].set_title("Polar Image with Shoreline")
    axs[0].set_xlabel("Azimuth Bin")
    axs[0].set_ylabel("Range Bin")

    if shoreline_points:
        axs[0].scatter(azs, ranges, c="red", s=5, label="Shoreline")
        axs[0].legend()

    # 2. Cartesian Image with Shoreline Overlay
    axs[1].imshow(
        cart_img,
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        origin="lower",
        cmap="viridis",
    )
    axs[1].set_title("Cartesian Image with Shoreline")
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Y (m)")

    if shoreline_points:
        axs[1].scatter(cart_shoreline_x, cart_shoreline_y, c="red", s=3, label="Shoreline", zorder=5)
        axs[1].legend()

    # 3. Cartesian Mask
    axs[2].imshow(
        cart_mask,
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        aspect="equal",
        cmap="gray",
        origin="lower",
    )
    axs[2].set_title("Cartesian Mask (Morph. Cleanup)")
    axs[2].set_xlabel("X (m)")

    # 4. Polar Mask Back-projected
    axs[3].imshow(
        polar_mask.T if polar_mask.shape[1] > polar_mask.shape[0] else polar_mask,
        aspect="auto",
        cmap="gray",
        origin="lower",
    )
    axs[3].set_title("Back-projected Polar Mask")
    axs[3].set_xlabel("Azimuth Bin")

    plt.tight_layout()
    _display_or_save("shoreline_test.png")
