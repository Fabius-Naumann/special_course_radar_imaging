import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from map.shoreline_registration import (
    Pose2D,
    ShorelineFrame,
    WeightedPointSet,
    _fetch_overpass_features,
    accumulate_shoreline_window,
    angle_diff_rad,
    build_coastline_map,
    register_shoreline_to_map,
    transform_points_world_to_local,
)


def _make_weighted_point_set(points: np.ndarray) -> WeightedPointSet:
    points = np.asarray(points, dtype=float)
    num_points = points.shape[0]
    return WeightedPointSet(
        points=points,
        weights=np.ones(num_points, dtype=float),
        persistence=np.full(num_points, 3.0, dtype=float),
        segment_ids=np.zeros(num_points, dtype=int),
        arc_lengths_m=np.zeros(num_points, dtype=float),
        quality_weights=np.ones(num_points, dtype=float),
    )


class ShorelineRegistrationTests(unittest.TestCase):
    @patch("map.shoreline_registration.requests.Session")
    def test_overpass_fetch_uses_form_encoded_data_field(self, session_cls):
        response = MagicMock()
        response.json.return_value = {"elements": []}
        response.raise_for_status.return_value = None

        session = MagicMock()
        session.post.return_value = response
        session_cls.return_value = session

        _fetch_overpass_features(55.7116, 12.5957, 2500.0, timeout_s=15.0)

        self.assertTrue(session.post.called)
        _, kwargs = session.post.call_args
        self.assertIn("data", kwargs)
        self.assertIsInstance(kwargs["data"], dict)
        self.assertIn("data", kwargs["data"])
        self.assertIn("[out:json]", kwargs["data"]["data"])
        self.assertIn('way["natural"="coastline"]', kwargs["data"]["data"])

    def test_registration_recovers_pose_with_outliers(self):
        rng = np.random.default_rng(7)
        coastline_map = build_coastline_map(
            center_latlon=(0.0, 0.0),
            radius_m=450.0,
            sample_step_m=5.0,
            local_features=[
                np.array([[-250.0, -80.0], [-180.0, -20.0], [-100.0, 30.0], [-20.0, 60.0], [60.0, 70.0], [140.0, 75.0]]),
                np.array([[80.0, -240.0], [60.0, -160.0], [35.0, -80.0], [25.0, 20.0], [30.0, 150.0]]),
                np.array([[-60.0, -180.0], [-20.0, -120.0], [20.0, -110.0], [90.0, -115.0]]),
            ],
            fetch_if_missing=False,
        )

        true_pose = Pose2D(x=32.0, y=-28.0, yaw_rad=np.deg2rad(38.0))
        local_points = transform_points_world_to_local(coastline_map.sampled_points, true_pose)
        visible_mask = (
            (local_points[:, 0] > 20.0)
            & (local_points[:, 0] < 240.0)
            & (np.abs(local_points[:, 1]) < 180.0)
        )
        shoreline_local = local_points[visible_mask][::3]
        shoreline_local = shoreline_local + rng.normal(scale=1.5, size=shoreline_local.shape)

        outliers = np.column_stack(
            (
                rng.uniform(30.0, 200.0, size=20),
                rng.uniform(-150.0, 150.0, size=20),
            )
        )
        point_set = _make_weighted_point_set(np.vstack((shoreline_local, outliers)))

        prior_pose = Pose2D(
            x=true_pose.x + 35.0,
            y=true_pose.y - 25.0,
            yaw_rad=true_pose.yaw_rad + np.deg2rad(60.0),
        )
        result = register_shoreline_to_map(
            point_set,
            prior_pose,
            coastline_map,
            translation_search_m=120.0,
            translation_step_m=12.0,
            rotation_search_deg=120.0,
            rotation_step_deg=10.0,
            min_inliers=20,
        )

        self.assertTrue(result.success, msg=result.rejection_reason)
        self.assertLess(np.hypot(result.estimated_pose.x - true_pose.x, result.estimated_pose.y - true_pose.y), 12.0)
        self.assertLess(abs(np.rad2deg(angle_diff_rad(result.estimated_pose.yaw_rad, true_pose.yaw_rad))), 8.0)
        self.assertGreater(result.confidence, 0.25)

    def test_temporal_accumulation_rejects_transient_outliers(self):
        rng = np.random.default_rng(11)
        shoreline = np.column_stack((np.linspace(60.0, 180.0, 30), 15.0 + 0.15 * np.linspace(60.0, 180.0, 30)))
        transient_boat = np.column_stack((np.linspace(90.0, 120.0, 8), np.linspace(-90.0, -70.0, 8)))

        frames = [
            ShorelineFrame(
                points=np.vstack((shoreline + rng.normal(scale=0.8, size=shoreline.shape), transient_boat)),
                latitude=55.7116,
                longitude=12.5957,
                heading_deg=92.0,
            ),
            ShorelineFrame(
                points=shoreline + rng.normal(scale=0.8, size=shoreline.shape),
                latitude=55.7116,
                longitude=12.5957,
                heading_deg=92.0,
            ),
            ShorelineFrame(
                points=shoreline + rng.normal(scale=0.8, size=shoreline.shape),
                latitude=55.7116,
                longitude=12.5957,
                heading_deg=92.0,
            ),
        ]

        accumulated = accumulate_shoreline_window(
            frames,
            anchor_index=1,
            cell_size_m=4.0,
            min_persistence=2,
            min_segment_points=4,
            min_segment_length_m=15.0,
        )

        self.assertGreater(accumulated.points.shape[0], 0)
        distances_to_boat = np.linalg.norm(accumulated.points - np.array([105.0, -80.0]), axis=1)
        self.assertTrue(np.all(distances_to_boat > 20.0))

    def test_straight_coastline_is_reported_as_weak(self):
        coastline_map = build_coastline_map(
            center_latlon=(0.0, 0.0),
            radius_m=500.0,
            sample_step_m=5.0,
            local_features=[np.array([[-320.0, 0.0], [320.0, 0.0]])],
            fetch_if_missing=False,
        )

        true_pose = Pose2D(x=25.0, y=65.0, yaw_rad=np.deg2rad(90.0))
        local_points = transform_points_world_to_local(coastline_map.sampled_points, true_pose)
        visible = local_points[(local_points[:, 0] > 10.0) & (local_points[:, 0] < 200.0)][::4]
        point_set = _make_weighted_point_set(visible)

        prior_pose = Pose2D(x=true_pose.x + 80.0, y=true_pose.y, yaw_rad=true_pose.yaw_rad)
        result = register_shoreline_to_map(
            point_set,
            prior_pose,
            coastline_map,
            translation_search_m=150.0,
            translation_step_m=10.0,
            rotation_search_deg=20.0,
            rotation_step_deg=5.0,
            min_inliers=15,
        )

        self.assertTrue((not result.success) or (result.observability != "good"))
        self.assertLess(result.confidence, 0.7)


if __name__ == "__main__":
    unittest.main()
