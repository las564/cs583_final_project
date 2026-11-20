"""PyTorch dataset for fixed-length pose sequences."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_TO_ID = {"bad": 0, "good": 1}
INPUT_CHANNELS = 3


def load_pose_index(index: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load and validate a pose manifest for training or evaluation."""
    frame = (
        index.copy()
        if isinstance(index, pd.DataFrame)
        else pd.read_csv(index, dtype={"subject_id": "string"})
    )
    required_columns = {"pose_path", "label", "split"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Pose index is missing required columns: {', '.join(sorted(missing_columns))}"
        )
    if frame.empty:
        raise ValueError("Pose index contains no records.")
    if frame["pose_path"].isna().any() or frame["label"].isna().any():
        raise ValueError("Pose index contains an empty pose_path or label.")
    frame["pose_path"] = frame["pose_path"].astype(str)
    frame["label"] = frame["label"].astype(str).str.lower()
    unknown_labels = set(frame["label"]).difference(LABEL_TO_ID)
    if unknown_labels:
        raise ValueError(
            "Expected labels 'good' and 'bad'; found "
            + ", ".join(sorted(unknown_labels))
        )
    frame["split"] = frame["split"].astype(str).str.lower()
    return frame


class PoseSequenceDataset(Dataset):
    """Load pose NPZ files and uniformly sample a fixed number of frames."""

    def __init__(
        self,
        records: pd.DataFrame,
        pose_root: str | Path,
        sequence_length: int = 64,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
        if records.empty:
            raise ValueError("Cannot create a pose dataset from zero records.")

        self.records = records.reset_index(drop=True)
        self.pose_root = Path(pose_root)
        self.sequence_length = sequence_length
        self.num_keypoints = self._inspect_first_pose()

    def _pose_path(self, pose_path: str) -> Path:
        path = Path(pose_path)
        return path if path.is_absolute() else self.pose_root / path

    def _inspect_first_pose(self) -> int:
        pose_path = self._pose_path(str(self.records.loc[0, "pose_path"]))
        if not pose_path.is_file():
            raise FileNotFoundError(f"Pose file does not exist: {pose_path}")
        with np.load(pose_path) as pose_file:
            keypoints = pose_file["keypoints"]
        if keypoints.ndim != 3 or keypoints.shape[2] < 4:
            raise ValueError(
                f"Expected keypoints with shape (frames, keypoints, 4+): {pose_path}"
            )
        return int(keypoints.shape[1])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records.iloc[index]
        pose_path = self._pose_path(str(record["pose_path"]))
        with np.load(pose_path) as pose_file:
            keypoints = pose_file["keypoints"]

        if keypoints.ndim != 3 or keypoints.shape[1] != self.num_keypoints:
            raise ValueError(f"Inconsistent keypoint count in pose file: {pose_path}")
        if keypoints.shape[2] < 4 or keypoints.shape[0] == 0:
            raise ValueError(f"Invalid or empty keypoint sequence: {pose_path}")

        # Use image-normalized x/y and confidence; depth is not available in YOLO.
        features = keypoints[..., [0, 1, 3]].astype(np.float32, copy=False)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        sample_indices = np.linspace(
            0,
            len(features) - 1,
            num=self.sequence_length,
        ).round().astype(np.int64)
        sampled_features = np.ascontiguousarray(features[sample_indices])

        label = LABEL_TO_ID[str(record["label"]).lower()]
        return (
            torch.from_numpy(sampled_features),
            torch.tensor(label, dtype=torch.long),
        )
