import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

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


def plot_shoreline_extraction(
    img,
    shoreline_points,
    cart_mask,
    polar_mask,
    cluster_labels=None,
    output_path=None,
    clockwise_azimuth=False,
):
    """
    Plot the results of shoreline extraction.
    """
    # Import locally to avoid circular imports
    import matplotlib.pyplot as plt
    import numpy as np

    from utils.data_loading import polar_to_cartesian_image, polar_to_cartesian_points

    cart_img, x_coords, y_coords = polar_to_cartesian_image(
        img,
        grid_size=1024,
        clockwise_azimuth=clockwise_azimuth,
    )

    cart_shoreline_x = []
    cart_shoreline_y = []
    if shoreline_points:
        azs, ranges = zip(*shoreline_points, strict=False)
        pts = polar_to_cartesian_points(
            np.array(ranges),
            np.array(azs),
            clockwise_azimuth=clockwise_azimuth,
        )
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


def local_xy_to_web_mercator(local_xy_m, center_lat, center_lon, heading_deg):
    """Transform local radar XY points (meters) into EPSG:3857 coordinates."""
    local_xy_m = np.asarray(local_xy_m, dtype=float)
    if local_xy_m.ndim != 2 or local_xy_m.shape[1] != 2:
        raise ValueError("local_xy_m must have shape (N, 2)")

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    center_x, center_y = to_merc.transform(float(center_lon), float(center_lat))

    heading_math_rad = np.radians(90.0 - float(heading_deg))
    c, s = np.cos(heading_math_rad), np.sin(heading_math_rad)
    rotation = np.array([[c, -s], [s, c]], dtype=float)

    rotated = local_xy_m @ rotation.T
    rotated[:, 0] += center_x
    rotated[:, 1] += center_y
    return rotated


def plot_radar_overlay_on_osm_static(
    center_lat,
    center_lon,
    heading_deg,
    output_path,
    shoreline_xy=None,
    cart_mask=None,
    max_range_m=1000.0,
    title="Radar Shoreline on OSM",
):
    """Plot radar shoreline and optional Cartesian mask on static OpenStreetMap basemap."""
    try:
        import contextily as ctx
    except ImportError as error:
        raise ImportError("contextily is required for static OSM plotting") from error

    fig, ax = plt.subplots(figsize=(9, 9))

    all_x = []
    all_y = []

    if cart_mask is not None:
        mask = np.asarray(cart_mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("cart_mask must be a 2D array")

        y_coords = np.linspace(-float(max_range_m), float(max_range_m), mask.shape[0], dtype=float)
        x_coords = np.linspace(-float(max_range_m), float(max_range_m), mask.shape[1], dtype=float)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        local_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        world_grid = local_xy_to_web_mercator(local_grid, center_lat, center_lon, heading_deg)
        world_x = world_grid[:, 0].reshape(mask.shape)
        world_y = world_grid[:, 1].reshape(mask.shape)

        ax.contourf(world_x, world_y, mask.astype(float), levels=[0.5, 1.5], alpha=0.45, colors=["royalblue"])
        all_x.extend([float(np.min(world_x)), float(np.max(world_x))])
        all_y.extend([float(np.min(world_y)), float(np.max(world_y))])

    if shoreline_xy is not None:
        shoreline_xy = np.asarray(shoreline_xy, dtype=float)
        if shoreline_xy.ndim == 2 and shoreline_xy.shape[0] > 0:
            shoreline_world = local_xy_to_web_mercator(shoreline_xy, center_lat, center_lon, heading_deg)
            ax.scatter(shoreline_world[:, 0], shoreline_world[:, 1], s=6, c="red", alpha=0.85)
            all_x.extend([float(np.min(shoreline_world[:, 0])), float(np.max(shoreline_world[:, 0]))])
            all_y.extend([float(np.min(shoreline_world[:, 1])), float(np.max(shoreline_world[:, 1]))])

    radar_center = local_xy_to_web_mercator(np.array([[0.0, 0.0]]), center_lat, center_lon, heading_deg)[0]
    ax.scatter(radar_center[0], radar_center[1], c="gold", s=60, marker="*", edgecolors="black", linewidths=0.5)
    all_x.append(float(radar_center[0]))
    all_y.append(float(radar_center[1]))

    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        pad = 60.0
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pose_arrow(ax, pose, color, label, arrow_length_m=60.0):
    direction = np.array([np.cos(float(pose.yaw_rad)), np.sin(float(pose.yaw_rad))], dtype=float)
    ax.arrow(
        float(pose.x),
        float(pose.y),
        float(direction[0] * arrow_length_m),
        float(direction[1] * arrow_length_m),
        width=1.5,
        head_width=8.0,
        head_length=10.0,
        color=color,
        length_includes_head=True,
        zorder=6,
    )
    ax.scatter([pose.x], [pose.y], c=color, s=36, label=label, zorder=7)


def _plot_covariance_ellipse(ax, center_xy, covariance_xy, color="black", n_std=2.0):
    covariance_xy = np.asarray(covariance_xy, dtype=float)
    if covariance_xy.shape != (2, 2) or not np.all(np.isfinite(covariance_xy)):
        return

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xy)
    if np.any(eigenvalues <= 0.0):
        return

    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width, height = 2.0 * n_std * np.sqrt(eigenvalues)

    ellipse = matplotlib.patches.Ellipse(
        xy=center_xy,
        width=float(width),
        height=float(height),
        angle=float(angle),
        edgecolor=color,
        facecolor="none",
        linewidth=1.5,
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(ellipse)


def plot_shoreline_registration_result(
    coastline_map,
    point_set,
    prior_pose,
    registration_result,
    output_path,
    title="Shoreline Registration",
):
    """Plot coastline registration diagnostics in the local map frame."""
    from map.shoreline_registration import transform_points_local_to_world

    point_set_points = np.asarray(point_set.points, dtype=float)
    prior_world = transform_points_local_to_world(point_set_points, prior_pose)
    corrected_world = np.asarray(registration_result.transformed_points, dtype=float)
    inlier_mask = np.asarray(registration_result.inlier_mask, dtype=bool)
    if inlier_mask.shape != (point_set_points.shape[0],):
        inlier_mask = np.zeros(point_set_points.shape[0], dtype=bool)

    fig, ax = plt.subplots(figsize=(9, 9))

    for line in coastline_map.local_lines:
        line = np.asarray(line, dtype=float)
        if line.ndim == 2 and line.shape[0] > 1:
            ax.plot(line[:, 0], line[:, 1], color="steelblue", linewidth=1.2, alpha=0.8, zorder=1)

    if coastline_map.sampled_points.shape[0] > 0:
        ax.scatter(
            coastline_map.sampled_points[:, 0],
            coastline_map.sampled_points[:, 1],
            s=4,
            c="lightsteelblue",
            alpha=0.35,
            zorder=1,
        )

    if prior_world.shape[0] > 0:
        ax.scatter(
            prior_world[:, 0],
            prior_world[:, 1],
            s=16,
            c="darkorange",
            alpha=0.45,
            label="Prior transformed points",
            zorder=2,
        )

    if corrected_world.shape[0] > 0:
        outlier_mask = ~inlier_mask
        if np.any(outlier_mask):
            ax.scatter(
                corrected_world[outlier_mask, 0],
                corrected_world[outlier_mask, 1],
                s=18,
                c="crimson",
                marker="x",
                linewidths=0.8,
                alpha=0.85,
                label="Rejected points",
                zorder=4,
            )
        if np.any(inlier_mask):
            ax.scatter(
                corrected_world[inlier_mask, 0],
                corrected_world[inlier_mask, 1],
                s=18,
                c="forestgreen",
                alpha=0.9,
                label="Inlier points",
                zorder=5,
            )

    _plot_pose_arrow(ax, prior_pose, color="darkorange", label="Prior pose")
    _plot_pose_arrow(ax, registration_result.estimated_pose, color="forestgreen", label="Corrected pose")
    if getattr(registration_result, "covariance", None) is not None:
        _plot_covariance_ellipse(
            ax,
            center_xy=(registration_result.estimated_pose.x, registration_result.estimated_pose.y),
            covariance_xy=np.asarray(registration_result.covariance, dtype=float)[:2, :2],
            color="black",
        )

    if getattr(registration_result, "coarse_hypotheses", None):
        coarse_positions = np.array(
            [[hyp.pose.x, hyp.pose.y] for hyp in registration_result.coarse_hypotheses],
            dtype=float,
        )
        ax.scatter(
            coarse_positions[:, 0],
            coarse_positions[:, 1],
            s=28,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            label="Coarse hypotheses",
            zorder=3,
        )

    text_lines = [
        f"success={registration_result.success}",
        f"confidence={registration_result.confidence:.2f}",
        f"inliers={registration_result.inlier_count}",
        f"inlier_ratio={registration_result.inlier_ratio:.2f}",
        f"mean_residual={registration_result.mean_abs_residual_m:.2f} m",
        f"observability={registration_result.observability}",
    ]
    if registration_result.rejection_reason:
        text_lines.append(f"reason={registration_result.rejection_reason}")

    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    ax.set_title(title)
    ax.set_xlabel("Local east (m)")
    ax.set_ylabel("Local north (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
