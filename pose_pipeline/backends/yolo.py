"""Ultralytics YOLO pose-estimation backend."""

import numpy as np

from ..base import missing_pose


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def smooth_pose_sequence(poses: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """EMA-smooth visible keypoint x/y coordinates without filling gaps."""
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("Smoothing alpha must be greater than 0 and at most 1.")
    if poses.ndim != 3 or poses.shape[2] < 4:
        raise ValueError("Expected poses with shape (frames, keypoints, 4+).")

    smoothed = poses.copy()
    for keypoint_index in range(poses.shape[1]):
        previous = None
        for frame_index in range(poses.shape[0]):
            coordinates = poses[frame_index, keypoint_index, :2]
            confidence = poses[frame_index, keypoint_index, 3]
            if confidence <= 0.0 or not np.isfinite(coordinates).all():
                previous = None
                continue

            if previous is not None:
                coordinates = alpha * coordinates + (1.0 - alpha) * previous
                smoothed[frame_index, keypoint_index, :2] = coordinates
            previous = coordinates.copy()
    return smoothed


class YOLOPoseEstimator:
    """Run a pretrained Ultralytics YOLO pose model on RGB frames."""

    backend_name = "yolo"
    keypoint_names = COCO_KEYPOINT_NAMES

    def __init__(
        self,
        model: str = "yolo11n-pose.pt",
        confidence: float = 0.25,
        smoothing_alpha: float = 0.4,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1.")
        if not np.isfinite(smoothing_alpha) or not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("YOLO smoothing alpha must be greater than 0 and at most 1.")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "The YOLO backend requires Ultralytics. Install it with "
                "`python3 -m pip install -r requirements-pose-yolo.txt`."
            ) from error

        self._model = YOLO(model)
        self._confidence = confidence
        self._smoothing_alpha = smoothing_alpha

    def estimate(self, rgb_frame: np.ndarray) -> np.ndarray:
        height, width = rgb_frame.shape[:2]
        results = self._model.predict(
            source=rgb_frame[..., ::-1].copy(),
            conf=self._confidence,
            verbose=False,
        )
        if not results or results[0].keypoints is None:
            return missing_pose(len(self.keypoint_names))

        result = results[0]
        data = result.keypoints.data.cpu().numpy()
        if data.shape[0] == 0:
            return missing_pose(len(self.keypoint_names))
        if data.shape[1] != len(self.keypoint_names):
            raise ValueError(
                "Expected YOLO COCO-17 keypoints, received "
                f"{data.shape[1]} keypoints."
            )

        confidences = (
            data[..., 2]
            if data.shape[2] >= 3
            else np.ones(data.shape[:2], dtype=np.float32)
        )
        boxes = result.boxes
        if boxes is not None and boxes.conf is not None and len(boxes.conf) == len(data):
            person_index = int(boxes.conf.argmax().item())
        else:
            person_index = int(confidences.mean(axis=1).argmax())

        pose = missing_pose(len(self.keypoint_names))
        pose[:, 0] = data[person_index, :, 0] / width
        pose[:, 1] = data[person_index, :, 1] / height
        pose[:, 3] = confidences[person_index]
        return pose

    def smooth_sequence(self, poses: np.ndarray) -> np.ndarray:
        return smooth_pose_sequence(poses, self._smoothing_alpha)

    def reset(self) -> None:
        """YOLO pose inference keeps no per-video tracking state."""

    def close(self) -> None:
        """The Ultralytics model has no persistent video handle to close."""
