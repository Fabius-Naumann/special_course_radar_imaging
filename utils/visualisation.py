import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils import RESULTS_DIR
else:
    from . import RESULTS_DIR


def _display_or_save(filename=None):
    """Display plot interactively or save it in headless environment."""
    backend = matplotlib.get_backend().lower()
    is_interactive = backend not in {"agg", "svg", "pdf", "ps", "cairo"}

    if is_interactive:
        plt.show()
    else:
        if filename is None:
            filename = "plot.png"
        filepath = RESULTS_DIR / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {filepath}")


def visualize_radar(
    polar_data=None,
    cartesian_data=None,
    x=None,
    y=None,
    data_layout="auto",
    mode="both",
):
    """
    Visualize radar data in polar and/or Cartesian coordinates.

    Parameters:
    -----------
    polar_data : ndarray, optional
        Radar data in polar format
    cartesian_data : ndarray, optional
        Radar data in Cartesian format
    x, y : ndarray, optional
        Cartesian coordinate arrays
    data_layout : str, optional
        'range_azimuth', 'azimuth_range', or 'auto'
    mode : str, optional
        'polar', 'cartesian', or 'both'
    """
    if mode not in {"polar", "cartesian", "both"}:
        raise ValueError("mode must be 'polar', 'cartesian', or 'both'")

    if mode in {"polar", "both"} and polar_data is None:
        raise ValueError("polar_data is required for mode='polar' or mode='both'")

    if mode in {"cartesian", "both"}:
        if cartesian_data is None or x is None or y is None:
            raise ValueError("cartesian_data, x, and y are required for mode='cartesian' or mode='both'")

    is_azimuth_range = polar_data is not None and (
        data_layout == "azimuth_range"
        or (data_layout == "auto" and polar_data.ndim == 2 and polar_data.shape[1] > polar_data.shape[0])
    )

    if is_azimuth_range:
        x_label, y_label = "Range (samples)", "Azimuth (samples)"
    else:
        x_label, y_label = "Azimuth (samples)", "Range (samples)"

    if mode == "polar":
        plt.figure(figsize=(8, 6))
        plt.imshow(polar_data, aspect="auto", cmap="viridis")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Radar Image (Polar)")
        plt.colorbar(label="Intensity")

    elif mode == "cartesian":
        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            cartesian_data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap="viridis",
        )
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Radar Image (Cartesian)")
        plt.gca().set_aspect("equal")
        plt.colorbar(im, label="Intensity")

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.imshow(polar_data, aspect="auto", cmap="viridis")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title("Radar Image (Polar)")

        im = ax2.imshow(
            cartesian_data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap="viridis",
        )
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title("Radar Image (Cartesian)")
        ax2.set_aspect("equal")
        plt.colorbar(im, ax=ax2, label="Intensity")

    plt.tight_layout()
    _display_or_save("radar_visualization.png")


def plot_radars_side_by_side(images, titles=None, fig_size=5, x=None, y=None, filename="radar_comparison.png"):
    """plot multiple input radar images side by side cartesian coordinates"""
    if x is None or y is None:
        x = y = np.linspace(-1000, 1000, images[0].shape[0], dtype=np.float32)
    if titles is None:
        titles = [f"Radar Image {i + 1}" for i in range(len(images))]
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(fig_size * num_images, fig_size))
    if num_images == 1:
        axes = [axes]

    for i, (ax, cartesian_data) in enumerate(zip(axes, images)):
        im = ax.imshow(
            cartesian_data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap="viridis",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(titles[i])
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

    plt.tight_layout()
    _display_or_save(filename)


def plot_shoreline_extraction(img, shoreline_points, cart_mask, polar_mask, cluster_labels=None, output_path=None):
    """
    Plot the results of shoreline extraction.
    """
    # Import locally to avoid circular imports
    import matplotlib.pyplot as plt
    import numpy as np

    from utils.data_loading import polar_to_cartesian_image, polar_to_cartesian_points

    cart_img, x_coords, y_coords = polar_to_cartesian_image(img, grid_size=1024)

    cart_shoreline_x = []
    cart_shoreline_y = []
    if shoreline_points:
        azs, ranges = zip(*shoreline_points)
        pts = polar_to_cartesian_points(np.array(ranges), np.array(azs))
        cart_shoreline_x = pts[:, 0]
        cart_shoreline_y = pts[:, 1]

    num_subplots = 5 if cluster_labels is not None else 4
    fig, axs = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))

    axs[0].imshow(img.T if img.shape[1] > img.shape[0] else img, aspect="auto", cmap="viridis", origin="lower")
    axs[0].set_title("Polar Image with Shoreline")
    axs[0].set_xlabel("Azimuth Bin")
    axs[0].set_ylabel("Range Bin")

    if shoreline_points:
        axs[0].scatter(azs, ranges, c="red", s=5, label="Shoreline")
        axs[0].legend()

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

    axs[2].imshow(
        cart_mask,
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        aspect="equal",
        cmap="gray",
        origin="lower",
    )
    axs[2].set_title("Cartesian Mask (Morph. Cleanup)")
    axs[2].set_xlabel("X (m)")

    axs[3].imshow(
        polar_mask.T if polar_mask.shape[1] > polar_mask.shape[0] else polar_mask,
        aspect="auto",
        cmap="gray",
        origin="lower",
    )
    axs[3].set_title("Back-projected Polar Mask")
    axs[3].set_xlabel("Azimuth Bin")

    if cluster_labels is not None:
        axs[4].imshow(
            cart_img,
            extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
            origin="lower",
            cmap="gray",
            alpha=0.5,
        )
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        axs[4].set_title(f"Shoreline Clusters (Estimated N={n_clusters})")
        axs[4].set_xlabel("X (m)")
        axs[4].set_ylabel("Y (m)")

        if len(cart_shoreline_x) > 0:
            cluster_labels = np.array(cluster_labels)
            outliers = cluster_labels == -1
            inliers = ~outliers
            if np.any(outliers):
                axs[4].scatter(
                    cart_shoreline_x[outliers],
                    cart_shoreline_y[outliers],
                    c="black",
                    s=4,
                    marker="x",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=4,
                )
            if np.any(inliers):
                axs[4].scatter(
                    cart_shoreline_x[inliers],
                    cart_shoreline_y[inliers],
                    c=cluster_labels[inliers],
                    cmap="tab20",
                    s=10,
                    zorder=5,
                )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        _display_or_save("shoreline_test.png")
