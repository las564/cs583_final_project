"""Render saved pose estimates over their source videos."""

import argparse
from pathlib import Path

import cv2
import numpy as np


SKELETON_EDGES = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
)


def _draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray,
    keypoint_names: tuple[str, ...],
    confidence_threshold: float,
) -> np.ndarray:
    if keypoints.shape != (len(keypoint_names), 4):
        raise ValueError(
            f"Expected pose shape ({len(keypoint_names)}, 4), got {keypoints.shape}."
        )

    height, width = frame.shape[:2]
    points = {}
    name_to_index = {name: index for index, name in enumerate(keypoint_names)}
    for name, index in name_to_index.items():
        x, y, _, confidence = keypoints[index]
        if (
            confidence >= confidence_threshold
            and np.isfinite(x)
            and np.isfinite(y)
        ):
            point = (
                int(np.clip(round(float(x) * width), 0, width - 1)),
                int(np.clip(round(float(y) * height), 0, height - 1)),
            )
            points[name] = point
            cv2.circle(frame, point, 4, (40, 220, 40), thickness=-1)

    for start_name, end_name in SKELETON_EDGES:
        if start_name in points and end_name in points:
            cv2.line(
                frame,
                points[start_name],
                points[end_name],
                (30, 180, 255),
                thickness=2,
            )
    return frame


def render_pose_video(
    video_path: str | Path,
    pose_path: str | Path,
    output_path: str | Path,
    confidence_threshold: float = 0.25,
) -> Path:
    """Write a copy of the source video with available poses overlaid."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    source_path = Path(video_path)
    pose_file_path = Path(pose_path)
    destination_path = Path(output_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("The rendered video must not overwrite the source video.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_path}")
    if not pose_file_path.is_file():
        raise FileNotFoundError(f"Pose file does not exist: {pose_file_path}")

    with np.load(pose_file_path) as pose_file:
        required_fields = {
            "keypoints",
            "frame_indices",
            "keypoint_names",
        }
        missing_fields = required_fields.difference(pose_file.files)
        if missing_fields:
            raise ValueError(
                f"Pose file is missing fields: {', '.join(sorted(missing_fields))}"
            )
        keypoints = pose_file["keypoints"]
        frame_indices = pose_file["frame_indices"]
        keypoint_names = tuple(str(name) for name in pose_file["keypoint_names"])

    if keypoints.ndim != 3 or keypoints.shape[2] < 4:
        raise ValueError("Pose keypoints must have shape (frames, keypoints, 4+).")
    if len(frame_indices) != len(keypoints) or len(keypoint_names) != keypoints.shape[1]:
        raise ValueError("Pose frame indices, names, and keypoint data do not align.")
    if len(set(frame_indices.tolist())) != len(frame_indices):
        raise ValueError("Pose frame_indices must be unique.")

    pose_by_frame = {
        int(frame_index): pose
        for frame_index, pose in zip(frame_indices, keypoints)
    }

    capture = cv2.VideoCapture(str(source_path))
    writer = None
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open source video: {source_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError(f"Could not read video dimensions: {source_path}")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination_path),
            cv2.VideoWriter.fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise OSError(f"Could not open output video for writing: {destination_path}")

        frame_index = 0
        written_frames = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            pose = pose_by_frame.get(frame_index)
            if pose is not None:
                frame = _draw_pose(
                    frame,
                    pose,
                    keypoint_names,
                    confidence_threshold,
                )
            writer.write(frame)
            frame_index += 1
            written_frames += 1

        if written_frames == 0:
            raise ValueError(f"Source video contains no readable frames: {source_path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return destination_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument("--pose", type=Path, required=True, help="Pose NPZ path.")
    parser.add_argument("--output", type=Path, required=True, help="Rendered MP4 path.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum keypoint confidence to draw (default: 0.25).",
    )
    args = parser.parse_args()
    result = render_pose_video(
        args.video,
        args.pose,
        args.output,
        confidence_threshold=args.confidence,
    )
    print(f"Wrote pose visualization to {result}")


if __name__ == "__main__":
    main()
