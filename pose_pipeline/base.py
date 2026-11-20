"""Shared interface for pose-estimation backends."""

from typing import Protocol

import numpy as np


class PoseEstimator(Protocol):
    """Backend contract: RGB frame in, normalized keypoints out."""

    backend_name: str
    keypoint_names: tuple[str, ...]

    def estimate(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Return (keypoints, x/y/z/confidence) for one frame."""
        ...

    def smooth_sequence(self, poses: np.ndarray) -> np.ndarray:
        """Apply backend-specific temporal filtering to one video's poses."""
        ...

    def reset(self) -> None:
        """Reset frame-tracking state before processing another video."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...


def missing_pose(keypoint_count: int) -> np.ndarray:
    """Represent a frame with no detected person."""
    pose = np.full((keypoint_count, 4), np.nan, dtype=np.float32)
    pose[:, 3] = 0.0
    return pose
