import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from scipy.ndimage import maximum_filter

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

    if mode in {"cartesian", "both"} and (cartesian_data is None or x is None or y is None):
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


def plot_registered_shoreline_overlay(
    coastline_map,
    registration_result,
    output_path,
    radar_overlay=None,
    radar_extent_m=1000.0,
    max_overlay_grid=800,
    overlay_color="orangered",
    overlay_alpha=0.45,
    overlay_cell_scale=1,
    title="Registered Shoreline Overlay",
):
    """Plot the registered radar layer on an OSM basemap using the registration map frame."""
    try:
        import contextily as ctx
    except ImportError as error:
        raise ImportError("contextily is required for static OSM overlay plotting") from error

    from map.shoreline_registration import transform_points_local_to_world

    if getattr(coastline_map, "transformer_to_geo", None) is None:
        raise ValueError("coastline_map.transformer_to_geo is required for OSM overlay plotting")

    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def map_local_to_web_mercator(points_xy):
        points_xy = np.asarray(points_xy, dtype=float)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy must have shape (N, 2)")
        lon, lat = coastline_map.transformer_to_geo.transform(points_xy[:, 0], points_xy[:, 1])
        merc_x, merc_y = to_mercator.transform(lon, lat)
        return np.column_stack((merc_x, merc_y))

    corrected_world = np.asarray(registration_result.transformed_points, dtype=float)
    inlier_mask = np.asarray(registration_result.inlier_mask, dtype=bool)
    if inlier_mask.shape != (corrected_world.shape[0],):
        inlier_mask = np.zeros(corrected_world.shape[0], dtype=bool)

    fig, ax = plt.subplots(figsize=(9, 9))
    all_x = []
    all_y = []

    for line in coastline_map.local_lines:
        line = np.asarray(line, dtype=float)
        if line.ndim == 2 and line.shape[0] > 1:
            line_web = map_local_to_web_mercator(line)
            ax.plot(line_web[:, 0], line_web[:, 1], color="deepskyblue", linewidth=1.4, alpha=0.9, zorder=4)
            all_x.extend([float(np.min(line_web[:, 0])), float(np.max(line_web[:, 0]))])
            all_y.extend([float(np.min(line_web[:, 1])), float(np.max(line_web[:, 1]))])

    if coastline_map.sampled_points.shape[0] > 0:
        sampled_web = map_local_to_web_mercator(coastline_map.sampled_points)
        ax.scatter(
            sampled_web[:, 0],
            sampled_web[:, 1],
            s=5,
            c="lightskyblue",
            alpha=0.25,
            label="OSM coastline samples",
            zorder=3,
        )
        all_x.extend([float(np.min(sampled_web[:, 0])), float(np.max(sampled_web[:, 0]))])
        all_y.extend([float(np.min(sampled_web[:, 1])), float(np.max(sampled_web[:, 1]))])

    if radar_overlay is not None:
        overlay = np.asarray(radar_overlay, dtype=float)
        if overlay.ndim == 3:
            overlay = np.nanmean(overlay[..., :3], axis=2)
        if overlay.ndim != 2:
            raise ValueError("radar_overlay must be a 2D image or an RGB/RGBA image")

        max_dim = max(overlay.shape)
        resolution_stride = max(1, int(np.ceil(max_dim / max(float(max_overlay_grid), 1.0))))
        if resolution_stride > 1:
            overlay = overlay[::resolution_stride, ::resolution_stride]

        cell_scale = max(1, int(round(float(overlay_cell_scale))))
        if cell_scale > 1:
            # Expand sparse detections visually without changing their map geometry.
            overlay = maximum_filter(overlay, size=cell_scale, mode="constant", cval=0.0)

        visible_overlay = np.isfinite(overlay) & (overlay > 0.0)
        if not np.any(visible_overlay):
            overlay = None
            visible_overlay = None
        else:
            finite_overlay = overlay[visible_overlay]
            low, high = np.percentile(finite_overlay, [2.0, 99.0])
            overlay = np.clip((overlay - low) / max(high - low, 1e-6), 0.0, 1.0)
            overlay = np.ma.masked_where(~visible_overlay, overlay)

    if radar_overlay is not None and overlay is not None:
        height, width = overlay.shape
        x_edges = np.linspace(-float(radar_extent_m), float(radar_extent_m), width + 1, dtype=float)
        y_edges = np.linspace(-float(radar_extent_m), float(radar_extent_m), height + 1, dtype=float)
        x_grid, y_grid = np.meshgrid(x_edges, y_edges)
        local_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        map_grid = transform_points_local_to_world(local_grid, registration_result.estimated_pose)
        web_grid = map_local_to_web_mercator(map_grid)
        world_x = web_grid[:, 0].reshape(y_grid.shape)
        world_y = web_grid[:, 1].reshape(y_grid.shape)
        red, green, blue, _ = matplotlib.colors.to_rgba(overlay_color)
        overlay_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "single_hue_overlay",
            [(red, green, blue, 0.0), (red, green, blue, float(overlay_alpha))],
        )

        ax.pcolormesh(
            world_x,
            world_y,
            overlay,
            cmap=overlay_cmap,
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            zorder=2,
            rasterized=True,
        )
        all_x.extend([float(np.nanmin(world_x)), float(np.nanmax(world_x))])
        all_y.extend([float(np.nanmin(world_y)), float(np.nanmax(world_y))])

    if corrected_world.shape[0] > 0:
        corrected_web = map_local_to_web_mercator(corrected_world)
        outlier_mask = ~inlier_mask
        if np.any(outlier_mask):
            ax.scatter(
                corrected_web[outlier_mask, 0],
                corrected_web[outlier_mask, 1],
                s=18,
                c="crimson",
                marker="x",
                linewidths=0.8,
                alpha=0.8,
                label="Rejected detections",
                zorder=4,
            )
        if np.any(inlier_mask):
            ax.scatter(
                corrected_web[inlier_mask, 0],
                corrected_web[inlier_mask, 1],
                s=18,
                c="forestgreen",
                alpha=0.9,
                label="Registered detections",
                zorder=5,
            )
        all_x.extend([float(np.min(corrected_web[:, 0])), float(np.max(corrected_web[:, 0]))])
        all_y.extend([float(np.min(corrected_web[:, 1])), float(np.max(corrected_web[:, 1]))])

    pose = registration_result.estimated_pose
    pose_line = np.array(
        [
            [float(pose.x), float(pose.y)],
            [
                float(pose.x) + 60.0 * np.cos(float(pose.yaw_rad)),
                float(pose.y) + 60.0 * np.sin(float(pose.yaw_rad)),
            ],
        ],
        dtype=float,
    )
    pose_web = map_local_to_web_mercator(pose_line)
    ax.arrow(
        pose_web[0, 0],
        pose_web[0, 1],
        pose_web[1, 0] - pose_web[0, 0],
        pose_web[1, 1] - pose_web[0, 1],
        width=1.5,
        head_width=8.0,
        head_length=10.0,
        color="forestgreen",
        length_includes_head=True,
        label="Registered pose",
        zorder=6,
    )
    ax.scatter(pose_web[0:1, 0], pose_web[0:1, 1], c="forestgreen", s=36, zorder=7)
    all_x.append(float(pose_web[0, 0]))
    all_y.append(float(pose_web[0, 1]))

    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        pad = 80.0
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)

    ax.set_title(title)
    ax.set_xlabel("Web Mercator east (m)")
    ax.set_ylabel("Web Mercator north (m)")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_registration_confidence_over_time(
    summary_rows,
    output_path,
    title="ICP Registration Confidence Over Time",
):
    """Plot per-frame ICP confidence using timestamps, with frame discretisation made explicit."""
    timestamps = []
    confidences = []
    residuals_m = []
    successes = []
    reasons = []

    for row in summary_rows:
        timestamp_value = row.get("timestamp")
        if not timestamp_value:
            continue
        try:
            timestamp = datetime.fromisoformat(str(timestamp_value))
        except ValueError:
            continue

        try:
            confidence = float(row.get("registration_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not np.isfinite(confidence):
            confidence = 0.0

        try:
            residual_m = float(row.get("registration_mean_residual_m", np.nan))
        except (TypeError, ValueError):
            residual_m = np.nan
        if not np.isfinite(residual_m):
            residual_m = np.nan

        timestamps.append(timestamp)
        confidences.append(float(np.clip(confidence, 0.0, 1.0)))
        residuals_m.append(residual_m)
        successes.append(bool(row.get("registration_success", False)))
        reasons.append(str(row.get("registration_reason", "")))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    residual_ax = ax.twinx()

    if timestamps:
        confidence_array = np.asarray(confidences, dtype=float)
        residual_array = np.asarray(residuals_m, dtype=float)
        success_mask = np.asarray(successes, dtype=bool)
        valid_residual = np.isfinite(residual_array)

        ax.vlines(
            timestamps,
            ymin=0.0,
            ymax=confidence_array,
            color="0.55",
            linewidth=0.8,
            alpha=0.45,
            label="Frame samples",
            zorder=1,
        )
        ax.step(
            timestamps,
            confidence_array,
            where="post",
            color="midnightblue",
            linewidth=1.8,
            label="Held confidence between frames",
            zorder=2,
        )

        if np.any(success_mask):
            ax.scatter(
                np.asarray(timestamps, dtype=object)[success_mask],
                confidence_array[success_mask],
                s=34,
                c="forestgreen",
                edgecolors="white",
                linewidths=0.5,
                label="Accepted registration",
                zorder=3,
            )
        if np.any(~success_mask):
            ax.scatter(
                np.asarray(timestamps, dtype=object)[~success_mask],
                confidence_array[~success_mask],
                s=42,
                c="crimson",
                marker="x",
                linewidths=1.1,
                label="Rejected / unavailable registration",
                zorder=3,
            )

        if np.any(valid_residual):
            residual_ax.step(
                np.asarray(timestamps, dtype=object)[valid_residual],
                residual_array[valid_residual],
                where="post",
                color="darkgoldenrod",
                linewidth=1.6,
                linestyle="--",
                label="Held ICP residual",
                zorder=2,
            )
            residual_ax.scatter(
                np.asarray(timestamps, dtype=object)[valid_residual],
                residual_array[valid_residual],
                s=24,
                c="darkgoldenrod",
                marker="s",
                edgecolors="white",
                linewidths=0.4,
                label="ICP residual samples",
                zorder=3,
            )

        failed_reasons = {}
        for success, reason in zip(successes, reasons, strict=False):
            if success or not reason:
                continue
            failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
        if failed_reasons:
            reason_text = "\n".join(
                f"{reason}: {count}" for reason, count in sorted(failed_reasons.items(), key=lambda item: item[0])[:6]
            )
            ax.text(
                0.99,
                0.04,
                reason_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.75"},
            )

        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.text(0.5, 0.5, "No timestamped registration rows", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Radar frame timestamp")
    ax.set_ylabel("ICP confidence")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0.0, 1.0, 11))
    residual_ax.set_ylabel("Mean absolute ICP residual (m)")
    residual_ax.set_ylim(bottom=0.0)
    residual_ax.tick_params(axis="y", colors="darkgoldenrod")
    residual_ax.yaxis.label.set_color("darkgoldenrod")
    residual_ax.grid(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.12)
    handles, labels = ax.get_legend_handles_labels()
    residual_handles, residual_labels = residual_ax.get_legend_handles_labels()
    ax.legend(handles + residual_handles, labels + residual_labels, loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
