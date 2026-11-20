"""MediaPipe Pose-estimation backend."""

import numpy as np

from ..base import missing_pose


class MediaPipePoseEstimator:
    """Run MediaPipe Pose on RGB frames."""

    backend_name = "mediapipe"

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise ImportError(
                "The MediaPipe backend requires MediaPipe. Install it with "
                "`python3 -m pip install -r requirements-pose-mediapipe.txt`."
            ) from error

        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "pose"):
            raise ImportError(
                "The installed MediaPipe package does not provide the solutions "
                "Pose API required by this backend."
            )
        pose_module = solutions.pose
        self._pose = pose_module.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.keypoint_names = tuple(
            landmark.name.lower() for landmark in pose_module.PoseLandmark
        )

    def estimate(self, rgb_frame: np.ndarray) -> np.ndarray:
        result = self._pose.process(rgb_frame)
        if result.pose_landmarks is None:
            return missing_pose(len(self.keypoint_names))

        pose = np.empty((len(self.keypoint_names), 4), dtype=np.float32)
        for index, landmark in enumerate(result.pose_landmarks.landmark):
            pose[index] = (
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
            )
        return pose

    def smooth_sequence(self, poses: np.ndarray) -> np.ndarray:
        """MediaPipe already smooths landmarks across frames by default."""
        return poses

    def reset(self) -> None:
        self._pose.reset()

    def close(self) -> None:
        self._pose.close()
