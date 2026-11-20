"""Command-line entry point for pose extraction."""

import argparse
from pathlib import Path

from .backends import create_pose_estimator
from .pipeline import run_pose_estimation


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Input CSV index.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repository_root / "data" / "raw",
        help="Root used to resolve relative video_path values.",
    )
    parser.add_argument(
        "--backend",
        choices=("yolo", "mediapipe"),
        default="yolo",
        help="Pose-estimation backend (default: yolo).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Ultralytics pose model or weights path (YOLO backend only).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.4,
        help="YOLO EMA smoothing alpha in (0, 1]; 1 disables smoothing (default: 0.4).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Estimate every Nth source frame (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for pose NPZ files (default: data/processed/poses/<backend>).",
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        help="Summary CSV path (default: <output-dir>/pose_index.csv).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also render each pose sequence over its source video.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        help="Directory for rendered videos (requires --visualize).",
    )
    args = parser.parse_args()

    if args.visualization_dir is not None and not args.visualize:
        parser.error("--visualization-dir requires --visualize.")

    output_dir = args.output_dir or (
        repository_root / "data" / "processed" / "poses" / args.backend
    )
    output_index = args.output_index or output_dir / "pose_index.csv"
    visualization_dir = None
    if args.visualize:
        visualization_dir = args.visualization_dir or (
            repository_root
            / "data"
            / "processed"
            / "visualizations"
            / args.backend
        )
    estimator = create_pose_estimator(
        args.backend,
        model=args.model,
        confidence=args.confidence,
        smoothing_alpha=args.smoothing_alpha,
    )
    try:
        result = run_pose_estimation(
            args.index,
            estimator,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            frame_stride=args.frame_stride,
            visualization_dir=visualization_dir,
        )
        output_index.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_index, index=False)
    finally:
        estimator.close()

    print(f"Wrote {len(result)} pose records to {output_index}")
    if visualization_dir is not None:
        print(f"Wrote visualizations to {visualization_dir}")
