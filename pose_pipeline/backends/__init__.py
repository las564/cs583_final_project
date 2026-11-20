"""Optional pose-estimation backends."""

from ..base import PoseEstimator
from .mediapipe import MediaPipePoseEstimator
from .yolo import YOLOPoseEstimator


def create_pose_estimator(
    backend: str,
    model: str = "yolo11n-pose.pt",
    confidence: float = 0.25,
    smoothing_alpha: float = 0.4,
) -> PoseEstimator:
    """Create one of the supported pose estimators.

    The MediaPipe backend ignores ``model``, ``confidence``, and
    ``smoothing_alpha``.
    """
    if backend == "yolo":
        return YOLOPoseEstimator(
            model=model,
            confidence=confidence,
            smoothing_alpha=smoothing_alpha,
        )
    if backend == "mediapipe":
        return MediaPipePoseEstimator()
    raise ValueError(f"Unknown pose backend {backend!r}; choose 'yolo' or 'mediapipe'.")


__all__ = [
    "MediaPipePoseEstimator",
    "YOLOPoseEstimator",
    "create_pose_estimator",
]
