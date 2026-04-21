from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, shape
from sklearn.cluster import DBSCAN


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def angle_diff_rad(angle_a: float, angle_b: float) -> float:
    """Smallest signed difference angle_a - angle_b."""
    return wrap_angle_rad(float(angle_a) - float(angle_b))


def compass_to_math_yaw_rad(heading_deg: float) -> float:
    """Convert compass heading (0 north, 90 east) to math yaw (0 east, +CCW)."""
    return np.deg2rad(90.0 - float(heading_deg))


def math_yaw_to_compass_deg(yaw_rad: float) -> float:
    """Convert math yaw (0 east, +CCW) to compass heading (0 north, 90 east)."""
    return float((90.0 - np.rad2deg(float(yaw_rad))) % 360.0)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_rad: float

    @property
    def yaw_deg(self) -> float:
        return float(np.rad2deg(self.yaw_rad))

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw_rad], dtype=float)


@dataclass
class WeightedPointSet:
    points: np.ndarray
    weights: np.ndarray
    persistence: np.ndarray
    segment_ids: np.ndarray
    arc_lengths_m: np.ndarray
    quality_weights: np.ndarray
    frame_count: int = 1
    anchor_pose: Pose2D | None = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.persistence = np.asarray(self.persistence, dtype=float)
        self.segment_ids = np.asarray(self.segment_ids, dtype=int)
        self.arc_lengths_m = np.asarray(self.arc_lengths_m, dtype=float)
        self.quality_weights = np.asarray(self.quality_weights, dtype=float)

        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        num_points = self.points.shape[0]
        expected_shapes = {
            "weights": self.weights.shape,
            "persistence": self.persistence.shape,
            "segment_ids": self.segment_ids.shape,
            "arc_lengths_m": self.arc_lengths_m.shape,
            "quality_weights": self.quality_weights.shape,
        }
        for name, value_shape in expected_shapes.items():
            if value_shape != (num_points,):
                raise ValueError(f"{name} must have shape ({num_points},)")

    def subset(self, mask: np.ndarray) -> "WeightedPointSet":
        mask = np.asarray(mask, dtype=bool)
        return WeightedPointSet(
            points=self.points[mask],
            weights=self.weights[mask],
            persistence=self.persistence[mask],
            segment_ids=self.segment_ids[mask],
            arc_lengths_m=self.arc_lengths_m[mask],
            quality_weights=self.quality_weights[mask],
            frame_count=self.frame_count,
            anchor_pose=self.anchor_pose,
        )


@dataclass
class ShorelineFrame:
    points: np.ndarray
    latitude: float
    longitude: float
    heading_deg: float
    weights: np.ndarray | None = None
    quality_weights: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    arc_lengths_m: np.ndarray | None = None
    frame_id: Any | None = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        num_points = self.points.shape[0]
        if self.weights is None:
            self.weights = np.ones(num_points, dtype=float)
        else:
            self.weights = np.asarray(self.weights, dtype=float)

        if self.quality_weights is None:
            self.quality_weights = np.asarray(self.weights, dtype=float)
        else:
            self.quality_weights = np.asarray(self.quality_weights, dtype=float)

        if self.segment_ids is None:
            self.segment_ids = np.full(num_points, -1, dtype=int)
        else:
            self.segment_ids = np.asarray(self.segment_ids, dtype=int)

        if self.arc_lengths_m is None:
            self.arc_lengths_m = np.zeros(num_points, dtype=float)
        else:
            self.arc_lengths_m = np.asarray(self.arc_lengths_m, dtype=float)

        for name, array in {
            "weights": self.weights,
            "quality_weights": self.quality_weights,
            "segment_ids": self.segment_ids,
            "arc_lengths_m": self.arc_lengths_m,
        }.items():
            if array.shape != (num_points,):
                raise ValueError(f"{name} must have shape ({num_points},)")


@dataclass
class CoastlineMap:
    center_lat: float
    center_lon: float
    radius_m: float
    sample_step_m: float
    grid_resolution_m: float
    local_lines: list[np.ndarray]
    sampled_points: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    segment_ids: np.ndarray
    arc_lengths_m: np.ndarray
    distance_transform_m: np.ndarray
    occupancy_mask: np.ndarray
    grid_extent_m: float
    transformer_to_local: Transformer | None = field(repr=False, default=None)
    transformer_to_geo: Transformer | None = field(repr=False, default=None)
    source_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.sampled_points = np.asarray(self.sampled_points, dtype=float)
        self.tangents = np.asarray(self.tangents, dtype=float)
        self.normals = np.asarray(self.normals, dtype=float)
        self.segment_ids = np.asarray(self.segment_ids, dtype=int)
        self.arc_lengths_m = np.asarray(self.arc_lengths_m, dtype=float)
        self.distance_transform_m = np.asarray(self.distance_transform_m, dtype=float)
        self.occupancy_mask = np.asarray(self.occupancy_mask, dtype=bool)

        if self.sampled_points.ndim != 2 or self.sampled_points.shape[1] != 2:
            raise ValueError("sampled_points must have shape (N, 2)")
        num_points = self.sampled_points.shape[0]
        for name, array in {
            "tangents": self.tangents,
            "normals": self.normals,
        }.items():
            if array.shape != (num_points, 2):
                raise ValueError(f"{name} must have shape ({num_points}, 2)")
        for name, array in {
            "segment_ids": self.segment_ids,
            "arc_lengths_m": self.arc_lengths_m,
        }.items():
            if array.shape != (num_points,):
                raise ValueError(f"{name} must have shape ({num_points},)")

        self.kdtree = cKDTree(self.sampled_points)


@dataclass(frozen=True)
class CoarseHypothesis:
    pose: Pose2D
    score: float
    residual_m: float


@dataclass
class RegistrationResult:
    success: bool
    estimated_pose: Pose2D
    prior_pose: Pose2D
    correction: Pose2D
    confidence: float
    observability: str
    covariance: np.ndarray
    condition_number: float
    mean_abs_residual_m: float
    inlier_count: int
    inlier_ratio: float
    coarse_hypotheses: list[CoarseHypothesis]
    coarse_score_gap: float
    inlier_mask: np.ndarray
    matched_indices: np.ndarray
    transformed_points: np.ndarray
    residuals_m: np.ndarray
    rejection_reason: str | None = None


def rotation_matrix(yaw_rad: float) -> np.ndarray:
    c = np.cos(float(yaw_rad))
    s = np.sin(float(yaw_rad))
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_points_local_to_world(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ rotation_matrix(pose.yaw_rad).T + np.array([pose.x, pose.y], dtype=float)


def transform_points_world_to_local(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (points - np.array([pose.x, pose.y], dtype=float)) @ rotation_matrix(pose.yaw_rad)


def _make_local_transformers(center_lat: float, center_lon: float) -> tuple[Transformer, Transformer]:
    local_crs = (
        f"+proj=aeqd +lat_0={float(center_lat)} +lon_0={float(center_lon)} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    to_geo = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    return to_local, to_geo


def pose_from_geodetic(
    latitude: float,
    longitude: float,
    heading_deg: float,
    center_lat: float,
    center_lon: float,
    *,
    transformer_to_local: Transformer | None = None,
) -> Pose2D:
    """Convert a geodetic pose to a local east-north pose."""
    if transformer_to_local is None:
        transformer_to_local, _ = _make_local_transformers(center_lat, center_lon)

    x_local, y_local = transformer_to_local.transform(float(longitude), float(latitude))
    return Pose2D(x=float(x_local), y=float(y_local), yaw_rad=compass_to_math_yaw_rad(float(heading_deg)))


def _coerce_pose(prior_pose: Pose2D | dict[str, Any] | tuple[float, float, float]) -> Pose2D:
    if isinstance(prior_pose, Pose2D):
        return prior_pose
    if isinstance(prior_pose, dict):
        yaw_rad = prior_pose.get("yaw_rad")
        if yaw_rad is None and "heading_deg" in prior_pose:
            yaw_rad = compass_to_math_yaw_rad(prior_pose["heading_deg"])
        if yaw_rad is None:
            raise ValueError("prior_pose dict must contain yaw_rad or heading_deg")
        return Pose2D(x=float(prior_pose["x"]), y=float(prior_pose["y"]), yaw_rad=float(yaw_rad))
    if len(prior_pose) != 3:
        raise ValueError("prior_pose must have 3 elements")
    return Pose2D(x=float(prior_pose[0]), y=float(prior_pose[1]), yaw_rad=float(prior_pose[2]))


def _iter_linear_geometries(geometry):
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, MultiLineString):
        for item in geometry.geoms:
            yield from _iter_linear_geometries(item)
        return
    if isinstance(geometry, Polygon):
        yield LineString(geometry.exterior.coords)
        for interior in geometry.interiors:
            yield LineString(interior.coords)
        return
    if isinstance(geometry, MultiPolygon):
        for item in geometry.geoms:
            yield from _iter_linear_geometries(item)
        return
    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from _iter_linear_geometries(item)


def _extract_features_from_geojson_payload(payload: dict[str, Any]) -> list[Any]:
    features: list[Any] = []
    if payload.get("type") == "FeatureCollection":
        items = payload.get("features", [])
    elif payload.get("type") == "Feature":
        items = [payload]
    else:
        items = [{"geometry": payload, "properties": {}}]

    for item in items:
        geometry_payload = item.get("geometry")
        if not geometry_payload:
            continue
        try:
            features.append(shape(geometry_payload))
        except Exception:
            continue
    return features


def _load_local_geometries(geojson_path: str | Path) -> list[Any]:
    with Path(geojson_path).open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    return _extract_features_from_geojson_payload(payload)


def _build_geojson_feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _overpass_bbox(center_lat: float, center_lon: float, radius_m: float) -> tuple[float, float, float, float]:
    lat_delta = float(radius_m) / 111_320.0
    lon_delta = float(radius_m) / (111_320.0 * max(np.cos(np.deg2rad(float(center_lat))), 1e-6))
    return (
        float(center_lat) - lat_delta,
        float(center_lon) - lon_delta,
        float(center_lat) + lat_delta,
        float(center_lon) + lon_delta,
    )


def _fetch_overpass_features(center_lat: float, center_lon: float, radius_m: float, timeout_s: float) -> list[dict[str, Any]]:
    south, west, north, east = _overpass_bbox(center_lat, center_lon, radius_m)
    query = f"""
    [out:json][timeout:{int(max(timeout_s, 10))}];
    (
      way["natural"="coastline"]({south},{west},{north},{east});
      way["natural"="water"]({south},{west},{north},{east});
      way["waterway"="riverbank"]({south},{west},{north},{east});
      way["water"="harbour"]({south},{west},{north},{east});
    );
    out geom;
    """.strip()

    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    payload = None
    request_errors = []
    session = requests.Session()
    headers = {
        "Accept": "application/json",
        "User-Agent": "special-course-radar-imaging/0.1 shoreline-registration",
    }

    for endpoint in endpoints:
        try:
            response = session.post(
                endpoint,
                data={"data": query},
                headers=headers,
                timeout=timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as error:
            request_errors.append(f"{endpoint}: {error}")

    if payload is None:
        joined_errors = "; ".join(request_errors) if request_errors else "unknown Overpass error"
        raise RuntimeError(f"Failed to fetch coastline data from Overpass: {joined_errors}")

    features: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry")
        if element.get("type") != "way" or not geometry or len(geometry) < 2:
            continue

        coordinates = [(float(node["lon"]), float(node["lat"])) for node in geometry]
        tags = dict(element.get("tags", {}))
        closed = len(coordinates) >= 4 and coordinates[0] == coordinates[-1]

        if closed and tags.get("natural") == "water":
            geometry_payload = {"type": "Polygon", "coordinates": [coordinates]}
        elif closed and tags.get("waterway") == "riverbank":
            geometry_payload = {"type": "Polygon", "coordinates": [coordinates]}
        elif closed and tags.get("water") == "harbour":
            geometry_payload = {"type": "Polygon", "coordinates": [coordinates]}
        else:
            geometry_payload = {"type": "LineString", "coordinates": coordinates}

        features.append(
            {
                "type": "Feature",
                "properties": tags,
                "geometry": geometry_payload,
            }
        )

    return features


def _sample_local_lines(
    local_lines: list[np.ndarray],
    sample_step_m: float,
    grid_extent_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampled_points: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    segment_ids: list[int] = []
    arc_lengths: list[float] = []

    for line_idx, coords in enumerate(local_lines):
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] < 2:
            continue

        cumulative = 0.0
        for start, end in zip(coords[:-1], coords[1:], strict=False):
            delta = end - start
            segment_length = float(np.linalg.norm(delta))
            if segment_length < 1e-6:
                continue

            tangent = delta / segment_length
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            num_steps = max(1, int(np.ceil(segment_length / max(float(sample_step_m), 1e-6))))
            ts = np.linspace(0.0, 1.0, num_steps, endpoint=False, dtype=float)
            ts = np.append(ts, 1.0)

            for t_value in ts:
                point = start + t_value * delta
                if np.max(np.abs(point)) > grid_extent_m + sample_step_m:
                    continue
                sampled_points.append(point)
                tangents.append(tangent)
                normals.append(normal)
                segment_ids.append(line_idx)
                arc_lengths.append(cumulative + t_value * segment_length)

            cumulative += segment_length

    if not sampled_points:
        raise ValueError("No coastline samples fall inside the requested local area.")

    return (
        np.asarray(sampled_points, dtype=float),
        np.asarray(tangents, dtype=float),
        np.asarray(normals, dtype=float),
        np.asarray(segment_ids, dtype=int),
        np.asarray(arc_lengths, dtype=float),
    )


def _build_distance_transform(
    sampled_points: np.ndarray,
    grid_extent_m: float,
    grid_resolution_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_size = int(np.ceil((2.0 * grid_extent_m) / grid_resolution_m)) + 1
    occupancy = np.zeros((grid_size, grid_size), dtype=bool)

    grid_x = np.round((sampled_points[:, 0] + grid_extent_m) / grid_resolution_m).astype(int)
    grid_y = np.round((sampled_points[:, 1] + grid_extent_m) / grid_resolution_m).astype(int)
    valid = (
        (grid_x >= 0)
        & (grid_x < grid_size)
        & (grid_y >= 0)
        & (grid_y < grid_size)
    )
    occupancy[grid_y[valid], grid_x[valid]] = True

    distance = distance_transform_edt(~occupancy) * grid_resolution_m
    return distance.astype(float), occupancy


def sample_distance_transform(coastline_map: CoastlineMap, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample the coastline distance transform in meters at arbitrary world-frame points."""
    points_world = np.asarray(points_world, dtype=float)
    distances = np.full(points_world.shape[0], coastline_map.grid_extent_m * 2.0, dtype=float)
    valid = np.zeros(points_world.shape[0], dtype=bool)

    cols = (points_world[:, 0] + coastline_map.grid_extent_m) / coastline_map.grid_resolution_m
    rows = (points_world[:, 1] + coastline_map.grid_extent_m) / coastline_map.grid_resolution_m
    valid = (
        (cols >= 0.0)
        & (cols <= coastline_map.distance_transform_m.shape[1] - 1)
        & (rows >= 0.0)
        & (rows <= coastline_map.distance_transform_m.shape[0] - 1)
    )

    if np.any(valid):
        sampled = map_coordinates(
            coastline_map.distance_transform_m,
            [rows[valid], cols[valid]],
            order=1,
            mode="nearest",
            prefilter=False,
        )
        distances[valid] = sampled.astype(float)

    return distances, valid


def build_coastline_map(
    center_latlon: tuple[float, float],
    radius_m: float,
    sample_step_m: float,
    *,
    geojson_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    local_features: list[np.ndarray] | None = None,
    fetch_if_missing: bool = True,
    grid_resolution_m: float | None = None,
    overpass_timeout_s: float = 30.0,
) -> CoastlineMap:
    """Build a sampled local coastline representation around the requested center."""
    center_lat, center_lon = float(center_latlon[0]), float(center_latlon[1])
    to_local, to_geo = _make_local_transformers(center_lat, center_lon)
    grid_resolution_m = float(grid_resolution_m or max(2.0, sample_step_m / 2.0))

    source_info: dict[str, Any] = {"mode": "unknown"}
    local_lines: list[np.ndarray] = []

    if local_features is not None:
        for coords in local_features:
            coords = np.asarray(coords, dtype=float)
            if coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] >= 2:
                local_lines.append(coords)
        source_info["mode"] = "local_features"
    else:
        geometries = []
        if geojson_path is not None:
            geometries = _load_local_geometries(geojson_path)
            source_info["mode"] = "geojson"
            source_info["path"] = str(geojson_path)
        elif cache_path is not None and Path(cache_path).exists():
            geometries = _load_local_geometries(cache_path)
            source_info["mode"] = "cache"
            source_info["path"] = str(cache_path)
        elif fetch_if_missing:
            features = _fetch_overpass_features(center_lat, center_lon, radius_m, timeout_s=overpass_timeout_s)
            geometries = _extract_features_from_geojson_payload(_build_geojson_feature_collection(features))
            source_info["mode"] = "overpass"
            source_info["feature_count"] = len(features)
            if cache_path is not None:
                cache_target = Path(cache_path)
                cache_target.parent.mkdir(parents=True, exist_ok=True)
                with cache_target.open("w", encoding="utf-8") as outfile:
                    json.dump(_build_geojson_feature_collection(features), outfile)
        else:
            raise FileNotFoundError("No coastline source was provided and fetch_if_missing=False")

        for geometry in geometries:
            for line in _iter_linear_geometries(geometry):
                coords_lonlat = np.asarray(line.coords, dtype=float)
                if coords_lonlat.shape[0] < 2:
                    continue
                xs, ys = to_local.transform(coords_lonlat[:, 0], coords_lonlat[:, 1])
                local_coords = np.column_stack((xs, ys)).astype(float)
                if np.all(np.max(np.abs(local_coords), axis=1) > radius_m * 1.5):
                    continue
                local_lines.append(local_coords)

    sampled_points, tangents, normals, segment_ids, arc_lengths = _sample_local_lines(
        local_lines=local_lines,
        sample_step_m=sample_step_m,
        grid_extent_m=radius_m,
    )
    distance_transform_m, occupancy_mask = _build_distance_transform(
        sampled_points=sampled_points,
        grid_extent_m=radius_m,
        grid_resolution_m=grid_resolution_m,
    )

    return CoastlineMap(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_m=float(radius_m),
        sample_step_m=float(sample_step_m),
        grid_resolution_m=float(grid_resolution_m),
        local_lines=local_lines,
        sampled_points=sampled_points,
        tangents=tangents,
        normals=normals,
        segment_ids=segment_ids,
        arc_lengths_m=arc_lengths,
        distance_transform_m=distance_transform_m,
        occupancy_mask=occupancy_mask,
        grid_extent_m=float(radius_m),
        transformer_to_local=to_local,
        transformer_to_geo=to_geo,
        source_info=source_info,
    )


def _coerce_frames(
    frames: list[ShorelineFrame | dict[str, Any]],
    prior_poses: list[Pose2D | dict[str, Any]] | None,
) -> list[ShorelineFrame]:
    shoreline_frames: list[ShorelineFrame] = []

    if prior_poses is not None and len(prior_poses) != len(frames):
        raise ValueError("prior_poses must match the number of frames")

    for index, frame in enumerate(frames):
        pose = prior_poses[index] if prior_poses is not None else None
        if isinstance(frame, ShorelineFrame):
            shoreline_frames.append(frame)
            continue

        if not isinstance(frame, dict):
            raise TypeError("frames must contain ShorelineFrame objects or dictionaries")

        if pose is not None:
            normalized_pose = _coerce_pose(pose)
            latitude = frame.get("latitude")
            longitude = frame.get("longitude")
            heading_deg = frame.get("heading_deg", math_yaw_to_compass_deg(normalized_pose.yaw_rad))
            if latitude is None or longitude is None:
                latitude = frame.get("anchor_latitude")
                longitude = frame.get("anchor_longitude")
            if latitude is None or longitude is None:
                raise ValueError("frame dictionaries need latitude/longitude when prior_poses are local")
        else:
            latitude = frame["latitude"]
            longitude = frame["longitude"]
            heading_deg = frame["heading_deg"]

        shoreline_frames.append(
            ShorelineFrame(
                points=np.asarray(frame["points"], dtype=float),
                latitude=float(latitude),
                longitude=float(longitude),
                heading_deg=float(heading_deg),
                weights=frame.get("weights"),
                quality_weights=frame.get("quality_weights"),
                segment_ids=frame.get("segment_ids"),
                arc_lengths_m=frame.get("arc_lengths_m"),
                frame_id=frame.get("frame_id", index),
            )
        )

    return shoreline_frames


def _cluster_anchor_points(
    points: np.ndarray,
    weights: np.ndarray,
    persistence: np.ndarray,
    *,
    cluster_eps_m: float,
    min_segment_points: int,
    min_segment_length_m: float,
) -> WeightedPointSet:
    if points.shape[0] == 0:
        return WeightedPointSet(
            points=np.empty((0, 2), dtype=float),
            weights=np.empty((0,), dtype=float),
            persistence=np.empty((0,), dtype=float),
            segment_ids=np.empty((0,), dtype=int),
            arc_lengths_m=np.empty((0,), dtype=float),
            quality_weights=np.empty((0,), dtype=float),
        )

    labels = DBSCAN(eps=cluster_eps_m, min_samples=max(2, min_segment_points // 2)).fit_predict(points)
    segment_ids = np.full(points.shape[0], -1, dtype=int)
    arc_lengths = np.zeros(points.shape[0], dtype=float)
    quality_weights = np.clip(weights, 0.0, 1.0)
    keep_mask = np.zeros(points.shape[0], dtype=bool)

    next_segment_id = 0
    for label_value in sorted(set(labels.tolist())):
        if label_value < 0:
            continue

        indices = np.where(labels == label_value)[0]
        if indices.size == 0:
            continue

        cluster_points = points[indices]
        if indices.size > 1:
            centered = cluster_points - np.mean(cluster_points, axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            principal_axis = vh[0]
            order = np.argsort(centered @ principal_axis)
            ordered_indices = indices[order]
            ordered_points = points[ordered_indices]
            diffs = np.linalg.norm(np.diff(ordered_points, axis=0), axis=1)
            segment_length_m = float(np.sum(diffs))
            arc = np.zeros(ordered_indices.size, dtype=float)
            arc[1:] = np.cumsum(diffs)
        else:
            ordered_indices = indices
            segment_length_m = 0.0
            arc = np.zeros(indices.size, dtype=float)

        if ordered_indices.size < min_segment_points or segment_length_m < min_segment_length_m:
            continue

        segment_ids[ordered_indices] = next_segment_id
        arc_lengths[ordered_indices] = arc
        segment_quality = min(1.0, segment_length_m / max(min_segment_length_m * 2.0, 1.0))
        quality_weights[ordered_indices] *= np.clip(0.5 + 0.5 * segment_quality, 0.5, 1.0)
        keep_mask[ordered_indices] = True
        next_segment_id += 1

    if not np.any(keep_mask):
        keep_mask = labels >= 0

    kept_points = points[keep_mask]
    kept_weights = np.clip(weights[keep_mask], 0.0, None)
    kept_persistence = persistence[keep_mask]
    kept_segment_ids = segment_ids[keep_mask]
    kept_arc_lengths = arc_lengths[keep_mask]
    kept_quality = quality_weights[keep_mask]

    return WeightedPointSet(
        points=kept_points,
        weights=kept_weights,
        persistence=kept_persistence,
        segment_ids=kept_segment_ids,
        arc_lengths_m=kept_arc_lengths,
        quality_weights=kept_quality,
    )


def accumulate_shoreline_window(
    frames: list[ShorelineFrame | dict[str, Any]],
    prior_poses: list[Pose2D | dict[str, Any]] | None = None,
    *,
    anchor_index: int | None = None,
    cell_size_m: float = 4.0,
    min_persistence: int = 2,
    cluster_eps_m: float = 12.0,
    min_segment_points: int = 5,
    min_segment_length_m: float = 20.0,
) -> WeightedPointSet:
    """Accumulate a short frame window into a single anchor-frame shoreline point set."""
    shoreline_frames = _coerce_frames(frames, prior_poses)
    if not shoreline_frames:
        raise ValueError("At least one frame is required")

    if anchor_index is None:
        anchor_index = len(shoreline_frames) // 2
    anchor_index = int(np.clip(anchor_index, 0, len(shoreline_frames) - 1))
    anchor_frame = shoreline_frames[anchor_index]

    to_local, _ = _make_local_transformers(anchor_frame.latitude, anchor_frame.longitude)
    anchor_pose_world = pose_from_geodetic(
        anchor_frame.latitude,
        anchor_frame.longitude,
        anchor_frame.heading_deg,
        center_lat=anchor_frame.latitude,
        center_lon=anchor_frame.longitude,
        transformer_to_local=to_local,
    )

    cell_stats: dict[tuple[int, int], dict[str, Any]] = {}
    for frame_index, frame in enumerate(shoreline_frames):
        frame_pose = pose_from_geodetic(
            frame.latitude,
            frame.longitude,
            frame.heading_deg,
            center_lat=anchor_frame.latitude,
            center_lon=anchor_frame.longitude,
            transformer_to_local=to_local,
        )
        world_points = transform_points_local_to_world(frame.points, frame_pose)
        anchor_points = transform_points_world_to_local(world_points, anchor_pose_world)
        point_weights = np.asarray(frame.weights, dtype=float) * np.asarray(frame.quality_weights, dtype=float)

        for point, weight in zip(anchor_points, point_weights, strict=False):
            key = tuple(np.floor(point / float(cell_size_m)).astype(int).tolist())
            stats = cell_stats.setdefault(
                key,
                {
                    "weighted_sum": np.zeros(2, dtype=float),
                    "weight_sum": 0.0,
                    "frames": set(),
                    "count": 0,
                },
            )
            stats["weighted_sum"] += float(max(weight, 1e-3)) * point
            stats["weight_sum"] += float(max(weight, 1e-3))
            stats["frames"].add(frame_index)
            stats["count"] += 1

    aggregated_points = []
    aggregated_weights = []
    aggregated_persistence = []
    for stats in cell_stats.values():
        persistence = len(stats["frames"])
        if persistence < int(min_persistence):
            continue
        point = stats["weighted_sum"] / max(stats["weight_sum"], 1e-6)
        weight = stats["weight_sum"] / max(stats["count"], 1)
        aggregated_points.append(point)
        aggregated_weights.append(weight * persistence)
        aggregated_persistence.append(float(persistence))

    aggregated = WeightedPointSet(
        points=np.asarray(aggregated_points, dtype=float).reshape(-1, 2)
        if aggregated_points
        else np.empty((0, 2), dtype=float),
        weights=np.asarray(aggregated_weights, dtype=float) if aggregated_weights else np.empty((0,), dtype=float),
        persistence=np.asarray(aggregated_persistence, dtype=float)
        if aggregated_persistence
        else np.empty((0,), dtype=float),
        segment_ids=np.full(len(aggregated_points), -1, dtype=int),
        arc_lengths_m=np.zeros(len(aggregated_points), dtype=float),
        quality_weights=np.asarray(aggregated_weights, dtype=float)
        if aggregated_weights
        else np.empty((0,), dtype=float),
        frame_count=len(shoreline_frames),
        anchor_pose=anchor_pose_world,
    )

    clustered = _cluster_anchor_points(
        aggregated.points,
        aggregated.weights,
        aggregated.persistence,
        cluster_eps_m=cluster_eps_m,
        min_segment_points=min_segment_points,
        min_segment_length_m=min_segment_length_m,
    )
    clustered.frame_count = len(shoreline_frames)
    clustered.anchor_pose = anchor_pose_world
    return clustered


def _coerce_point_set(points: WeightedPointSet | dict[str, Any] | np.ndarray) -> WeightedPointSet:
    if isinstance(points, WeightedPointSet):
        return points
    if isinstance(points, dict):
        point_array = np.asarray(points["points"], dtype=float)
        num_points = point_array.shape[0]
        return WeightedPointSet(
            points=point_array,
            weights=np.asarray(points.get("weights", np.ones(num_points)), dtype=float),
            persistence=np.asarray(points.get("persistence", np.ones(num_points)), dtype=float),
            segment_ids=np.asarray(points.get("segment_ids", np.full(num_points, -1)), dtype=int),
            arc_lengths_m=np.asarray(points.get("arc_lengths_m", np.zeros(num_points)), dtype=float),
            quality_weights=np.asarray(points.get("quality_weights", np.ones(num_points)), dtype=float),
            frame_count=int(points.get("frame_count", 1)),
            anchor_pose=points.get("anchor_pose"),
        )
    point_array = np.asarray(points, dtype=float)
    num_points = point_array.shape[0]
    return WeightedPointSet(
        points=point_array,
        weights=np.ones(num_points, dtype=float),
        persistence=np.ones(num_points, dtype=float),
        segment_ids=np.full(num_points, -1, dtype=int),
        arc_lengths_m=np.zeros(num_points, dtype=float),
        quality_weights=np.ones(num_points, dtype=float),
    )


def _select_spatially_separated_hypotheses(
    candidates: list[CoarseHypothesis],
    top_k: int,
    separation_m: float,
    separation_deg: float,
) -> list[CoarseHypothesis]:
    selected: list[CoarseHypothesis] = []
    separation_rad = np.deg2rad(float(separation_deg))
    for hypothesis in sorted(candidates, key=lambda item: item.score):
        too_close = False
        for chosen in selected:
            distance = np.hypot(
                hypothesis.pose.x - chosen.pose.x,
                hypothesis.pose.y - chosen.pose.y,
            )
            yaw_distance = abs(angle_diff_rad(hypothesis.pose.yaw_rad, chosen.pose.yaw_rad))
            if distance < separation_m and yaw_distance < separation_rad:
                too_close = True
                break
        if too_close:
            continue
        selected.append(hypothesis)
        if len(selected) >= top_k:
            break
    return selected


def _coarse_search(
    point_set: WeightedPointSet,
    prior_pose: Pose2D,
    coastline_map: CoastlineMap,
    *,
    translation_search_m: float,
    translation_step_m: float,
    rotation_search_deg: float,
    rotation_step_deg: float,
    top_k: int,
    distance_clip_m: float,
    prior_translation_sigma_m: float,
    prior_rotation_sigma_deg: float,
    hypothesis_separation_m: float,
    hypothesis_separation_deg: float,
    trim_fraction: float = 0.65,
) -> list[CoarseHypothesis]:
    if point_set.points.shape[0] == 0:
        return []

    dx_values = np.arange(-translation_search_m, translation_search_m + 0.5 * translation_step_m, translation_step_m)
    dy_values = np.arange(-translation_search_m, translation_search_m + 0.5 * translation_step_m, translation_step_m)
    dtheta_values_deg = np.arange(
        -rotation_search_deg,
        rotation_search_deg + 0.5 * rotation_step_deg,
        rotation_step_deg,
    )

    weights = np.clip(point_set.weights * point_set.quality_weights, 1e-3, None)
    candidates: list[CoarseHypothesis] = []
    prior_rotation_sigma_rad = np.deg2rad(float(prior_rotation_sigma_deg))

    for dtheta_deg in dtheta_values_deg:
        yaw = wrap_angle_rad(prior_pose.yaw_rad + np.deg2rad(dtheta_deg))
        rotated = point_set.points @ rotation_matrix(yaw).T
        for dx in dx_values:
            for dy in dy_values:
                pose = Pose2D(
                    x=prior_pose.x + float(dx),
                    y=prior_pose.y + float(dy),
                    yaw_rad=yaw,
                )
                transformed = rotated + np.array([pose.x, pose.y], dtype=float)
                distances, valid = sample_distance_transform(coastline_map, transformed)
                if not np.any(valid):
                    continue

                clipped = np.minimum(distances, float(distance_clip_m))
                valid_distances = clipped[valid]
                valid_weights = weights[valid]
                keep_count = max(10, int(np.ceil(trim_fraction * valid_distances.shape[0])))
                keep_order = np.argsort(valid_distances)[:keep_count]
                mean_residual = float(np.average(valid_distances[keep_order], weights=valid_weights[keep_order]))
                score = float(np.average(valid_distances[keep_order] ** 2, weights=valid_weights[keep_order]))
                score += (float(dx) / max(prior_translation_sigma_m, 1e-6)) ** 2
                score += (float(dy) / max(prior_translation_sigma_m, 1e-6)) ** 2
                score += (np.deg2rad(float(dtheta_deg)) / max(prior_rotation_sigma_rad, 1e-6)) ** 2
                candidates.append(CoarseHypothesis(pose=pose, score=score, residual_m=mean_residual))

    return _select_spatially_separated_hypotheses(
        candidates=candidates,
        top_k=top_k,
        separation_m=hypothesis_separation_m,
        separation_deg=hypothesis_separation_deg,
    )


def _huber_weights(abs_residual: np.ndarray, delta_m: float) -> np.ndarray:
    weights = np.ones_like(abs_residual, dtype=float)
    large = abs_residual > float(delta_m)
    weights[large] = float(delta_m) / np.maximum(abs_residual[large], 1e-6)
    return weights


def _run_trimmed_point_to_line_icp(
    point_set: WeightedPointSet,
    prior_pose: Pose2D,
    initial_pose: Pose2D,
    coastline_map: CoastlineMap,
    *,
    max_iterations: int,
    gate_m: float,
    trim_fraction: float,
    huber_delta_m: float,
    prior_translation_sigma_m: float,
    prior_rotation_sigma_deg: float,
    min_inliers: int,
) -> dict[str, Any]:
    pose = initial_pose.as_vector().copy()
    prior_vector = prior_pose.as_vector()
    prior_lambdas = np.array(
        [
            1.0 / max(float(prior_translation_sigma_m) ** 2, 1e-6),
            1.0 / max(float(prior_translation_sigma_m) ** 2, 1e-6),
            1.0 / max(np.deg2rad(float(prior_rotation_sigma_deg)) ** 2, 1e-6),
        ],
        dtype=float,
    )

    num_points = point_set.points.shape[0]
    final_inlier_mask = np.zeros(num_points, dtype=bool)
    final_match_indices = np.full(num_points, -1, dtype=int)
    final_residuals = np.full(num_points, np.nan, dtype=float)
    final_transformed = transform_points_local_to_world(point_set.points, Pose2D(*pose))
    final_hessian = np.diag(prior_lambdas)

    for _ in range(max_iterations):
        current_pose = Pose2D(x=float(pose[0]), y=float(pose[1]), yaw_rad=float(pose[2]))
        transformed = transform_points_local_to_world(point_set.points, current_pose)
        distances, indices = coastline_map.kdtree.query(transformed, k=1, distance_upper_bound=float(gate_m))
        valid = np.isfinite(distances) & (indices < coastline_map.sampled_points.shape[0])
        if int(np.sum(valid)) < int(min_inliers):
            break

        valid_indices = np.where(valid)[0]
        matched_indices = indices[valid]
        matched_points = coastline_map.sampled_points[matched_indices]
        matched_normals = coastline_map.normals[matched_indices]
        residuals = np.einsum("ij,ij->i", transformed[valid] - matched_points, matched_normals)

        keep_count = max(int(min_inliers), int(np.ceil(trim_fraction * residuals.shape[0])))
        keep_order = np.argsort(np.abs(residuals))[:keep_count]
        selected_points_idx = valid_indices[keep_order]
        selected_match_idx = matched_indices[keep_order]
        selected_normals = matched_normals[keep_order]
        selected_residuals = residuals[keep_order]
        selected_world = transformed[selected_points_idx]
        selected_weights = np.clip(
            point_set.weights[selected_points_idx] * point_set.quality_weights[selected_points_idx],
            1e-3,
            None,
        )
        robust_weights = _huber_weights(np.abs(selected_residuals), delta_m=huber_delta_m)
        total_weights = selected_weights * robust_weights

        world_relative = selected_world - pose[:2]
        dwdtheta = np.column_stack((-world_relative[:, 1], world_relative[:, 0]))
        jacobian = np.column_stack(
            (
                selected_normals[:, 0],
                selected_normals[:, 1],
                np.einsum("ij,ij->i", selected_normals, dwdtheta),
            )
        )

        weighted_jacobian = jacobian * total_weights[:, None]
        hessian = weighted_jacobian.T @ jacobian
        gradient = weighted_jacobian.T @ selected_residuals

        prior_delta = pose - prior_vector
        prior_delta[2] = angle_diff_rad(pose[2], prior_vector[2])
        hessian += np.diag(prior_lambdas)
        gradient += prior_lambdas * prior_delta

        try:
            step = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(hessian) @ gradient

        pose[:2] += step[:2]
        pose[2] = wrap_angle_rad(pose[2] + step[2])

        final_transformed = transformed
        final_inlier_mask[:] = False
        final_inlier_mask[selected_points_idx] = True
        final_match_indices[:] = -1
        final_match_indices[selected_points_idx] = selected_match_idx
        final_residuals[:] = np.nan
        final_residuals[selected_points_idx] = selected_residuals
        final_hessian = hessian

        if float(np.linalg.norm(step[:2])) < 0.05 and abs(float(step[2])) < np.deg2rad(0.1):
            break

    estimated_pose = Pose2D(x=float(pose[0]), y=float(pose[1]), yaw_rad=float(pose[2]))
    inlier_count = int(np.sum(final_inlier_mask))
    if inlier_count > 0:
        mean_abs_residual = float(np.nanmean(np.abs(final_residuals[final_inlier_mask])))
    else:
        mean_abs_residual = float("inf")

    return {
        "estimated_pose": estimated_pose,
        "inlier_mask": final_inlier_mask,
        "matched_indices": final_match_indices,
        "transformed_points": transform_points_local_to_world(point_set.points, estimated_pose),
        "residuals_m": final_residuals,
        "hessian": final_hessian,
        "mean_abs_residual_m": mean_abs_residual,
        "inlier_count": inlier_count,
    }


def register_shoreline_to_map(
    points: WeightedPointSet | dict[str, Any] | np.ndarray,
    prior_pose: Pose2D | dict[str, Any] | tuple[float, float, float],
    coastline_map: CoastlineMap,
    *,
    translation_search_m: float = 150.0,
    translation_step_m: float = 15.0,
    rotation_search_deg: float = 180.0,
    rotation_step_deg: float = 10.0,
    top_k: int = 5,
    coarse_distance_clip_m: float = 40.0,
    hypothesis_separation_m: float = 30.0,
    hypothesis_separation_deg: float = 20.0,
    icp_max_iterations: int = 25,
    icp_gate_m: float = 25.0,
    icp_trim_fraction: float = 0.7,
    icp_huber_delta_m: float = 6.0,
    prior_translation_sigma_m: float = 120.0,
    prior_rotation_sigma_deg: float = 45.0,
    min_inliers: int = 25,
    min_inlier_ratio: float = 0.35,
    max_mean_residual_m: float = 12.0,
    min_coarse_score_gap: float = 0.05,
    max_condition_number: float = 1e5,
) -> RegistrationResult:
    """Register anchor-frame shoreline points against a local coastline map."""
    point_set = _coerce_point_set(points)
    prior_pose = _coerce_pose(prior_pose)

    if point_set.points.shape[0] == 0:
        return RegistrationResult(
            success=False,
            estimated_pose=prior_pose,
            prior_pose=prior_pose,
            correction=Pose2D(0.0, 0.0, 0.0),
            confidence=0.0,
            observability="unobservable",
            covariance=np.full((3, 3), np.nan),
            condition_number=float("inf"),
            mean_abs_residual_m=float("inf"),
            inlier_count=0,
            inlier_ratio=0.0,
            coarse_hypotheses=[],
            coarse_score_gap=0.0,
            inlier_mask=np.zeros(0, dtype=bool),
            matched_indices=np.zeros(0, dtype=int),
            transformed_points=np.empty((0, 2), dtype=float),
            residuals_m=np.empty((0,), dtype=float),
            rejection_reason="no_points",
        )

    coarse_hypotheses = _coarse_search(
        point_set=point_set,
        prior_pose=prior_pose,
        coastline_map=coastline_map,
        translation_search_m=translation_search_m,
        translation_step_m=translation_step_m,
        rotation_search_deg=rotation_search_deg,
        rotation_step_deg=rotation_step_deg,
        top_k=top_k,
        distance_clip_m=coarse_distance_clip_m,
        prior_translation_sigma_m=prior_translation_sigma_m,
        prior_rotation_sigma_deg=prior_rotation_sigma_deg,
        hypothesis_separation_m=hypothesis_separation_m,
        hypothesis_separation_deg=hypothesis_separation_deg,
    )

    if not coarse_hypotheses:
        return RegistrationResult(
            success=False,
            estimated_pose=prior_pose,
            prior_pose=prior_pose,
            correction=Pose2D(0.0, 0.0, 0.0),
            confidence=0.0,
            observability="unobservable",
            covariance=np.full((3, 3), np.nan),
            condition_number=float("inf"),
            mean_abs_residual_m=float("inf"),
            inlier_count=0,
            inlier_ratio=0.0,
            coarse_hypotheses=[],
            coarse_score_gap=0.0,
            inlier_mask=np.zeros(point_set.points.shape[0], dtype=bool),
            matched_indices=np.full(point_set.points.shape[0], -1, dtype=int),
            transformed_points=transform_points_local_to_world(point_set.points, prior_pose),
            residuals_m=np.full(point_set.points.shape[0], np.nan),
            rejection_reason="no_coarse_hypothesis",
        )

    if len(coarse_hypotheses) >= 2:
        coarse_score_gap = float(
            (coarse_hypotheses[1].score - coarse_hypotheses[0].score)
            / max(abs(coarse_hypotheses[0].score), 1e-6)
        )
    else:
        coarse_score_gap = 1.0

    refined_candidates = []
    for hypothesis in coarse_hypotheses:
        refined = _run_trimmed_point_to_line_icp(
            point_set=point_set,
            prior_pose=prior_pose,
            initial_pose=hypothesis.pose,
            coastline_map=coastline_map,
            max_iterations=icp_max_iterations,
            gate_m=icp_gate_m,
            trim_fraction=icp_trim_fraction,
            huber_delta_m=icp_huber_delta_m,
            prior_translation_sigma_m=prior_translation_sigma_m,
            prior_rotation_sigma_deg=prior_rotation_sigma_deg,
            min_inliers=min_inliers,
        )
        refined["coarse_hypothesis"] = hypothesis
        refined_candidates.append(refined)

    refined_candidates.sort(
        key=lambda item: (
            item["mean_abs_residual_m"],
            -item["inlier_count"],
        )
    )
    best = refined_candidates[0]
    estimated_pose = best["estimated_pose"]
    inlier_count = int(best["inlier_count"])
    inlier_ratio = inlier_count / max(point_set.points.shape[0], 1)
    condition_number = float(np.linalg.cond(best["hessian"])) if np.all(np.isfinite(best["hessian"])) else float("inf")

    try:
        covariance = np.linalg.pinv(best["hessian"])
    except np.linalg.LinAlgError:
        covariance = np.full((3, 3), np.nan)

    observability = "good"
    if condition_number > max_condition_number:
        observability = "weak"
    if not np.all(np.isfinite(covariance)):
        observability = "unobservable"

    success = True
    rejection_reason = None
    if inlier_count < min_inliers:
        success = False
        rejection_reason = "too_few_inliers"
    elif inlier_ratio < min_inlier_ratio:
        success = False
        rejection_reason = "low_inlier_ratio"
    elif best["mean_abs_residual_m"] > max_mean_residual_m:
        success = False
        rejection_reason = "high_residual"
    elif coarse_score_gap < min_coarse_score_gap:
        success = False
        rejection_reason = "ambiguous_coarse_match"
    elif condition_number > max_condition_number:
        success = False
        rejection_reason = "weak_observability"

    confidence = 1.0
    confidence *= np.exp(-best["mean_abs_residual_m"] / max(max_mean_residual_m, 1e-6))
    confidence *= min(1.0, inlier_ratio / max(min_inlier_ratio, 1e-6))
    confidence *= min(1.0, coarse_score_gap / max(min_coarse_score_gap * 2.0, 1e-6))
    confidence *= min(1.0, max_condition_number / max(condition_number, 1.0))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if not success:
        confidence *= 0.5

    correction = Pose2D(
        x=float(estimated_pose.x - prior_pose.x),
        y=float(estimated_pose.y - prior_pose.y),
        yaw_rad=angle_diff_rad(estimated_pose.yaw_rad, prior_pose.yaw_rad),
    )

    return RegistrationResult(
        success=success,
        estimated_pose=estimated_pose,
        prior_pose=prior_pose,
        correction=correction,
        confidence=confidence,
        observability=observability,
        covariance=covariance,
        condition_number=condition_number,
        mean_abs_residual_m=float(best["mean_abs_residual_m"]),
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        coarse_hypotheses=coarse_hypotheses,
        coarse_score_gap=coarse_score_gap,
        inlier_mask=np.asarray(best["inlier_mask"], dtype=bool),
        matched_indices=np.asarray(best["matched_indices"], dtype=int),
        transformed_points=np.asarray(best["transformed_points"], dtype=float),
        residuals_m=np.asarray(best["residuals_m"], dtype=float),
        rejection_reason=rejection_reason,
    )
