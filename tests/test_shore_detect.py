import unittest

import numpy as np

from map.shore_detect import (
    _build_shoreline_metadata,
    cluster_shoreline_dbscan,
    cluster_shoreline_points,
)
from utils.data_loading import polar_to_cartesian_points


class ShorelineSegmentationTests(unittest.TestCase):
    def test_azimuth_continuity_keeps_sparse_far_range_chain(self):
        n_azimuth_bins = 400
        range_resolution_m = 1.0
        shoreline_points = [(azimuth, 1000) for azimuth in range(20, 42, 2)]
        azimuths, ranges = zip(*shoreline_points, strict=False)
        cart_points = polar_to_cartesian_points(
            np.asarray(ranges, dtype=float),
            np.asarray(azimuths, dtype=float),
            range_resolution=range_resolution_m,
            angle_resolution=2.0 * np.pi / n_azimuth_bins,
        )

        dbscan_labels = cluster_shoreline_dbscan(
            cart_points[:, 0],
            cart_points[:, 1],
            min_samples=3,
            cut_distance=10.0,
        )
        continuity_labels = cluster_shoreline_points(
            shoreline_points,
            cart_points=cart_points,
            n_azimuth_bins=n_azimuth_bins,
            range_resolution_m=range_resolution_m,
            method="azimuth_continuity",
            continuity_max_azimuth_gap_bins=3,
        )

        self.assertTrue(np.all(dbscan_labels == -1))
        self.assertEqual(set(continuity_labels.tolist()), {0})

    def test_sparse_far_range_chain_can_be_marked_valid_metadata(self):
        n_azimuth_bins = 400
        range_resolution_m = 1.0
        shoreline_points = [(azimuth, 1000) for azimuth in range(20, 42, 2)]
        azimuths, ranges = zip(*shoreline_points, strict=False)
        cart_points = polar_to_cartesian_points(
            np.asarray(ranges, dtype=float),
            np.asarray(azimuths, dtype=float),
            range_resolution=range_resolution_m,
            angle_resolution=2.0 * np.pi / n_azimuth_bins,
        )
        labels = cluster_shoreline_points(
            shoreline_points,
            cart_points=cart_points,
            n_azimuth_bins=n_azimuth_bins,
            range_resolution_m=range_resolution_m,
        )

        metadata = _build_shoreline_metadata(
            shoreline_points,
            cart_points,
            labels,
            min_segment_points=5,
            min_segment_length_m=100.0,
        )

        self.assertTrue(np.all(metadata["valid_mask"]))
        self.assertGreater(float(np.min(metadata["quality_weights"])), 0.0)


if __name__ == "__main__":
    unittest.main()
