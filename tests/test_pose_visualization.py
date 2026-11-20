import unittest

import numpy as np

from pose_pipeline.visualize import _draw_pose


class PoseVisualizationTests(unittest.TestCase):
    def test_draws_confident_keypoints_and_known_skeleton_edges(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        keypoints = np.asarray(
            [
                [0.25, 0.5, np.nan, 0.9],
                [0.75, 0.5, np.nan, 0.9],
            ],
            dtype=np.float32,
        )

        rendered = _draw_pose(
            frame,
            keypoints,
            ("left_shoulder", "right_shoulder"),
            confidence_threshold=0.25,
        )

        self.assertTrue(np.any(rendered[8, 4]))
        self.assertTrue(np.any(rendered[8, 12]))
        self.assertTrue(np.any(rendered[8, 8]))


if __name__ == "__main__":
    unittest.main()
