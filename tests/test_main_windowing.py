import unittest

from map.main import _window_indices


class MainWindowingTests(unittest.TestCase):
    def test_causal_window_uses_current_and_past_only(self):
        self.assertEqual(_window_indices(7, 20, 5), [3, 4, 5, 6, 7])

    def test_single_frame_window_keeps_current_frame(self):
        self.assertEqual(_window_indices(7, 20, 1), [7])

    def test_future_window_is_explicit_and_symmetric_for_odd_sizes(self):
        self.assertEqual(_window_indices(7, 20, 5, include_future=True), [5, 6, 7, 8, 9])

    def test_windows_are_clamped_to_available_images(self):
        self.assertEqual(_window_indices(1, 4, 5), [0, 1])
        self.assertEqual(_window_indices(1, 4, 5, include_future=True), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
