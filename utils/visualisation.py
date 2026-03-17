import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from __init__ import RESULTS_DIR


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
