"""Create a CSV index of the exercise videos in the raw dataset."""

import argparse
import math
import random
import re
from pathlib import Path

import pandas as pd


BASE_FIELDNAMES = (
    "video_path",
    "exercise",
    "label",
    "viewpoint",
    "subject_id",
)
SUBJECT_PATTERN = re.compile(r"subject_(\d+)_")
VIDEO_SUFFIXES = {".mp4"}


def create_index(dataset_root: Path) -> pd.DataFrame:
    """Return one metadata row per video, with paths relative to dataset_root."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    rows = []
    for video_path in sorted(dataset_root.rglob("*")):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_SUFFIXES:
            continue

        relative_path = video_path.relative_to(dataset_root)
        if len(relative_path.parts) != 4:
            raise ValueError(
                "Expected videos at <exercise>/<label>/<viewpoint>/<filename>: "
                f"{relative_path}"
            )

        subject_match = SUBJECT_PATTERN.match(video_path.name)
        if subject_match is None:
            raise ValueError(f"Could not parse subject ID from filename: {video_path.name}")

        exercise, label, viewpoint, _ = relative_path.parts
        rows.append(
            {
                "video_path": relative_path.as_posix(),
                "exercise": exercise,
                "label": label,
                "viewpoint": viewpoint,
                "subject_id": subject_match.group(1),
            }
        )

    if not rows:
        raise ValueError(f"No supported video files found in {dataset_root}")
    return pd.DataFrame.from_records(rows, columns=BASE_FIELDNAMES)


def create_participant_split(
    rows: pd.DataFrame,
    exercise: str,
    viewpoint: str,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Filter to one exercise/viewpoint and assign whole subjects to splits."""
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must all be positive and sum to 1.")

    selected_rows = rows.loc[
        rows["exercise"].eq(exercise) & rows["viewpoint"].eq(viewpoint)
    ].copy()
    if selected_rows.empty:
        raise ValueError(
            f"No videos found for exercise={exercise!r}, viewpoint={viewpoint!r}"
        )

    subjects = sorted(selected_rows["subject_id"].unique().tolist())
    if len(subjects) < 3:
        raise ValueError(
            "At least 3 participants are required to create train/validation/test splits."
        )

    remaining = len(subjects) - 3
    exact_counts = [remaining * ratio for ratio in ratios]
    split_counts = [1 + math.floor(count) for count in exact_counts]
    leftover = len(subjects) - sum(split_counts)
    remainder_order = sorted(
        range(len(ratios)),
        key=lambda index: exact_counts[index] - math.floor(exact_counts[index]),
        reverse=True,
    )
    for index in remainder_order[:leftover]:
        split_counts[index] += 1

    random.Random(seed).shuffle(subjects)
    split_names = ("train", "validation", "test")
    subject_splits = {}
    offset = 0
    for split_name, count in zip(split_names, split_counts):
        for subject_id in subjects[offset : offset + count]:
            subject_splits[subject_id] = split_name
        offset += count

    selected_rows["split"] = selected_rows["subject_id"].map(subject_splits)
    return selected_rows


def write_index(rows: pd.DataFrame, output_path: Path) -> None:
    """Write index rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)


def default_output_path(
    repository_root: Path, exercise: str | None = None, viewpoint: str | None = None
) -> Path:
    """Choose the default CSV name for a full index or filtered split."""
    metadata_directory = repository_root / "data" / "metadata"
    if exercise is not None and viewpoint is not None:
        return metadata_directory / f"{exercise}_{viewpoint}_split_index.csv"
    return metadata_directory / "dataset_index.csv"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repository_root / "data" / "raw",
        help="Root containing <exercise>/<label>/<viewpoint> video folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV path (defaults to an exercise/view split name or dataset_index.csv).",
    )
    parser.add_argument(
        "--exercise",
        help="Exercise to include. Must be used with --viewpoint to create splits.",
    )
    parser.add_argument(
        "--viewpoint",
        help="Viewpoint to include. Must be used with --exercise to create splits.",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.7, 0.15, 0.15),
        help="Participant split ratios (default: 0.7 0.15 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for participant assignment (default: 42).",
    )
    args = parser.parse_args()

    if bool(args.exercise) != bool(args.viewpoint):
        parser.error("--exercise and --viewpoint must be provided together.")

    rows = create_index(args.dataset_root)
    if args.exercise and args.viewpoint:
        rows = create_participant_split(
            rows,
            args.exercise,
            args.viewpoint,
            tuple(args.split_ratios),
            args.seed,
        )
    output_path = args.output or default_output_path(
        repository_root, args.exercise, args.viewpoint
    )
    write_index(rows, output_path)
    print(f"Wrote {len(rows)} video records to {output_path}")


if __name__ == "__main__":
    main()
