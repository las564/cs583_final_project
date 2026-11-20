import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pose_pipeline import (
    MediaPipePoseEstimator,
    YOLOPoseEstimator,
    load_index,
    run_pose_estimation,
)
from pose_pipeline.backends.yolo import smooth_pose_sequence
from pose_pipeline import pipeline as pipeline_module


class FakePoseEstimator:
    backend_name = "fake"
    keypoint_names: tuple[str, ...] = ("hip", "shoulder")

    def __init__(self) -> None:
        self.smooth_calls = 0

    def estimate(self, rgb_frame: np.ndarray) -> np.ndarray:
        if rgb_frame.ndim != 3:
            raise ValueError("Expected an RGB image frame.")
        return np.asarray(
            [
                [0.25, 0.5, np.nan, 0.9],
                [0.75, 0.2, np.nan, 0.8],
            ],
            dtype=np.float32,
        )

    def reset(self) -> None:
        pass

    def smooth_sequence(self, poses: np.ndarray) -> np.ndarray:
        self.smooth_calls += 1
        return poses

    def close(self) -> None:
        pass


class PosePipelineTests(unittest.TestCase):
    def test_load_index_accepts_dataframe_and_csv_and_preserves_subject_ids(self):
        source = pd.DataFrame(
            {
                "video_path": ["squat/good/front/subject_001.mp4"],
                "subject_id": ["001"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "index.csv"
            source.to_csv(csv_path, index=False)

            loaded_csv = load_index(csv_path)
            loaded_frame = load_index(source)

        self.assertEqual(loaded_csv.loc[0, "subject_id"], "001")
        self.assertEqual(loaded_frame.loc[0, "video_path"], source.loc[0, "video_path"])

    def test_run_pose_estimation_saves_npz_and_returns_manifest_dataframe(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25.0
        capture.read.side_effect = [
            (True, frame),
            (True, frame),
            (False, None),
        ]
        index = pd.DataFrame(
            {
                "video_path": ["squat/good/front/subject_001.mp4"],
                "exercise": ["squat"],
                "label": ["good"],
                "viewpoint": ["front"],
                "subject_id": ["001"],
                "split": ["train"],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "squat/good/front/subject_001.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.touch()
            output_dir = root / "poses"
            estimator = FakePoseEstimator()

            with (
                patch.object(
                    pipeline_module.cv2, "VideoCapture", return_value=capture
                ),
                patch.object(
                    pipeline_module,
                    "tqdm",
                    side_effect=lambda iterable, **kwargs: iterable,
                ) as progress,
                patch("pose_pipeline.visualize.render_pose_video") as render,
            ):
                result = run_pose_estimation(
                    index,
                    estimator,
                    dataset_root=root,
                    output_dir=output_dir,
                    visualization_dir=root / "visualizations",
                )
            progress.assert_called_once()
            self.assertEqual(
                progress.call_args.kwargs,
                {
                    "total": 1,
                    "desc": "Extracting poses",
                    "unit": "video",
                },
            )
            render.assert_called_once_with(
                video_path.resolve(),
                output_dir / "squat/good/front/subject_001.npz",
                root / "visualizations" / "squat/good/front/subject_001.mp4",
            )
            self.assertEqual(
                result.loc[0, "visualization_path"],
                "squat/good/front/subject_001.mp4",
            )
            self.assertEqual(estimator.smooth_calls, 1)

            self.assertEqual(result.loc[0, "subject_id"], "001")
            self.assertEqual(result.loc[0, "split"], "train")
            self.assertEqual(result.loc[0, "pose_backend"], "fake")
            self.assertEqual(result.loc[0, "pose_frame_count"], 2)
            self.assertEqual(result.loc[0, "fps"], 25.0)

            pose_path = Path(str(result.loc[0, "pose_path"]))
            with np.load(output_dir / pose_path) as pose_file:
                self.assertEqual(pose_file["keypoints"].shape, (2, 2, 4))
                np.testing.assert_array_equal(pose_file["frame_indices"], [0, 1])
                np.testing.assert_allclose(
                    pose_file["frame_times_seconds"], [0.0, 0.04]
                )
                self.assertEqual(
                    pose_file["keypoint_names"].tolist(), ["hip", "shoulder"]
                )

        capture.release.assert_called_once()

    def test_relative_video_path_cannot_escape_dataset_root(self):
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            pipeline_module._resolve_video_path("../outside.mp4", Path("/dataset"))

    def test_pose_pipeline_requires_video_path_column(self):
        with self.assertRaisesRegex(ValueError, "video_path"):
            load_index(pd.DataFrame({"exercise": ["squat"]}))

    def test_yolo_backend_uses_highest_confidence_person_and_normalizes_xy(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def argmax(self):
                return SimpleNamespace(item=lambda: int(self.value.argmax()))

            def __len__(self):
                return len(self.value)

        data = np.zeros((2, 17, 3), dtype=np.float32)
        data[0, :, :] = [10.0, 20.0, 0.4]
        data[1, :, :] = [100.0, 50.0, 0.8]
        result = SimpleNamespace(
            keypoints=SimpleNamespace(data=FakeTensor(data)),
            boxes=SimpleNamespace(conf=FakeTensor([0.1, 0.9])),
        )
        model = SimpleNamespace(predict=Mock(return_value=[result]))
        fake_ultralytics = SimpleNamespace(YOLO=Mock(return_value=model))

        with patch.dict("sys.modules", {"ultralytics": fake_ultralytics}):
            estimator = YOLOPoseEstimator()
            rgb_frame = np.zeros((100, 200, 3), dtype=np.uint8)
            rgb_frame[0, 0] = [10, 20, 30]
            pose = estimator.estimate(rgb_frame)

        self.assertEqual(pose.shape, (17, 4))
        np.testing.assert_allclose(pose[0, :2], [0.5, 0.5])
        self.assertTrue(np.isnan(pose[0, 2]))
        self.assertAlmostEqual(float(pose[0, 3]), 0.8)
        np.testing.assert_array_equal(
            model.predict.call_args.kwargs["source"][0, 0],
            [30, 20, 10],
        )

    def test_yolo_smoothing_filters_xy_and_does_not_fill_missing_keypoints(self):
        poses = np.asarray(
            [
                [[0.0, 0.0, np.nan, 0.9]],
                [[1.0, 1.0, np.nan, 0.9]],
                [[np.nan, np.nan, np.nan, 0.0]],
                [[0.8, 0.8, np.nan, 0.9]],
            ],
            dtype=np.float32,
        )

        smoothed = smooth_pose_sequence(poses, alpha=0.5)

        np.testing.assert_allclose(smoothed[:2, 0, :2], [[0.0, 0.0], [0.5, 0.5]])
        self.assertTrue(np.isnan(smoothed[2, 0, :2]).all())
        np.testing.assert_allclose(smoothed[3, 0, :2], [0.8, 0.8])
        np.testing.assert_array_equal(smoothed[:, 0, 3], poses[:, 0, 3])

    def test_mediapipe_backend_maps_named_landmarks_to_common_shape(self):
        class FakeLandmarkName(Enum):
            HIP = 0
            SHOULDER = 1

        fake_pose = SimpleNamespace(
            process=Mock(
                return_value=SimpleNamespace(
                    pose_landmarks=SimpleNamespace(
                        landmark=[
                            SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9),
                            SimpleNamespace(x=0.4, y=0.5, z=0.6, visibility=0.8),
                        ]
                    )
                )
            ),
            reset=Mock(),
            close=Mock(),
        )
        pose_module = SimpleNamespace(
            PoseLandmark=FakeLandmarkName,
            Pose=Mock(return_value=fake_pose),
        )
        fake_mediapipe = SimpleNamespace(
            solutions=SimpleNamespace(pose=pose_module)
        )

        with patch.dict("sys.modules", {"mediapipe": fake_mediapipe}):
            estimator = MediaPipePoseEstimator()
            pose = estimator.estimate(np.zeros((8, 8, 3), dtype=np.uint8))
            estimator.reset()
            estimator.close()

        self.assertEqual(estimator.keypoint_names, ("hip", "shoulder"))
        np.testing.assert_allclose(pose[0], [0.1, 0.2, 0.3, 0.9])
        fake_pose.reset.assert_called_once()
        fake_pose.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
