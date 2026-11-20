"""Index loading and video-to-pose extraction."""

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .base import PoseEstimator


def load_index(index: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load a CSV index or copy a DataFrame and validate its video paths."""
    if isinstance(index, pd.DataFrame):
        frame = index.copy()
    else:
        frame = pd.read_csv(index, dtype={"subject_id": "string"})

    if "video_path" not in frame.columns:
        raise ValueError("The index must contain a 'video_path' column.")
    if frame.empty:
        raise ValueError("The index contains no video records.")
    if frame["video_path"].isna().any():
        raise ValueError("The index contains an empty video_path.")

    frame["video_path"] = frame["video_path"].astype(str)
    if frame["video_path"].str.strip().eq("").any():
        raise ValueError("The index contains an empty video_path.")
    if frame["video_path"].duplicated().any():
        raise ValueError("The index contains duplicate video_path values.")
    return frame


def _resolve_video_path(video_path: str, dataset_root: Path) -> tuple[Path, Path]:
    path = Path(video_path)
    resolved_root = dataset_root.resolve()
    resolved_video = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()
    try:
        relative_path = resolved_video.relative_to(resolved_root)
    except ValueError:
        if not path.is_absolute():
            raise ValueError(
                f"Relative video_path escapes dataset root: {video_path}"
            ) from None
        digest = hashlib.sha256(str(resolved_video).encode("utf-8")).hexdigest()[:12]
        relative_path = Path(f"{resolved_video.stem}_{digest}{resolved_video.suffix}")
    return resolved_video, relative_path


def _extract_video(
    video_path: Path,
    estimator: PoseEstimator,
    frame_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has an invalid frame rate ({fps}): {video_path}")

        poses = []
        frame_indices = []
        frame_times = []
        source_frame_count = 0
        while True:
            success, bgr_frame = capture.read()
            if not success:
                break

            frame_index = source_frame_count
            source_frame_count += 1
            if frame_index % frame_stride:
                continue

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pose = np.asarray(estimator.estimate(rgb_frame), dtype=np.float32)
            expected_shape = (len(estimator.keypoint_names), 4)
            if pose.shape != expected_shape:
                raise ValueError(
                    f"{estimator.backend_name} returned pose shape {pose.shape}; "
                    f"expected {expected_shape}."
                )
            poses.append(pose)
            frame_indices.append(frame_index)
            frame_times.append(frame_index / fps)

        if source_frame_count == 0:
            raise ValueError(f"Video contains no readable frames: {video_path}")
    finally:
        capture.release()

    return (
        np.stack(poses),
        np.asarray(frame_indices, dtype=np.int64),
        np.asarray(frame_times, dtype=np.float32),
        fps,
        source_frame_count,
    )


def run_pose_estimation(
    index: pd.DataFrame | str | Path,
    estimator: PoseEstimator,
    dataset_root: str | Path,
    output_dir: str | Path,
    frame_stride: int = 1,
    visualization_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Extract pose sequences and optionally render annotated videos."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")

    rows = load_index(index)
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    visualization_root = (
        Path(visualization_dir) if visualization_dir is not None else None
    )

    if not estimator.keypoint_names:
        raise ValueError("The pose estimator must provide keypoint_names.")

    results = []
    for _, row in rows.iterrows():
        video_path, relative_video_path = _resolve_video_path(row["video_path"], root)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        estimator.reset()
        poses, frame_indices, frame_times, fps, source_frame_count = _extract_video(
            video_path, estimator, frame_stride
        )
        poses = np.asarray(estimator.smooth_sequence(poses), dtype=np.float32)
        if poses.shape != (len(frame_indices), len(estimator.keypoint_names), 4):
            raise ValueError(
                f"{estimator.backend_name} returned smoothed poses with invalid "
                f"shape: {poses.shape}."
            )
        pose_relative_path = relative_video_path.with_suffix(".npz")
        pose_path = output_root / pose_relative_path
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pose_path,
            keypoints=poses,
            frame_indices=frame_indices,
            frame_times_seconds=frame_times,
            keypoint_names=np.asarray(estimator.keypoint_names, dtype=str),
        )

        result_row = row.to_dict()
        result_row.update(
            {
                "pose_path": pose_relative_path.as_posix(),
                "pose_backend": estimator.backend_name,
                "pose_frame_count": len(frame_indices),
                "source_frame_count": source_frame_count,
                "fps": fps,
                "frame_stride": frame_stride,
            }
        )
        if visualization_root is not None:
            from .visualize import render_pose_video

            visualization_relative_path = relative_video_path.with_suffix(".mp4")
            render_pose_video(
                video_path,
                pose_path,
                visualization_root / visualization_relative_path,
            )
            result_row["visualization_path"] = (
                visualization_relative_path.as_posix()
            )
        results.append(result_row)

    return pd.DataFrame.from_records(results)
