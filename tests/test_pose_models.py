import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.classifiers import (
    CNNLSTMPoseClassifier,
    FullyConnectedPoseClassifier,
)
from models.dataset import PoseSequenceDataset
from models.training import (
    evaluate_pose_classifier,
    train_pose_classifier,
)


class PoseModelTests(unittest.TestCase):
    def test_both_classifiers_return_binary_logits(self):
        sequence = torch.randn(2, 8, 17, 3)

        for model in (
            FullyConnectedPoseClassifier(num_keypoints=17, sequence_length=8),
            CNNLSTMPoseClassifier(num_keypoints=17, sequence_length=8),
        ):
            logits = model(sequence)
            self.assertEqual(tuple(logits.shape), (2, 2))

    def test_dataset_resamples_and_replaces_missing_features(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            pose_root = Path(temporary_directory)
            keypoints = np.asarray(
                [
                    [[0.1, 0.2, np.nan, 0.9]],
                    [[np.nan, np.nan, np.nan, 0.0]],
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                pose_root / "pose.npz",
                keypoints=keypoints,
            )
            records = pd.DataFrame(
                {"pose_path": ["pose.npz"], "label": ["good"]}
            )

            dataset = PoseSequenceDataset(records, pose_root, sequence_length=4)
            sequence, label = dataset[0]

        self.assertEqual(tuple(sequence.shape), (4, 1, 3))
        self.assertEqual(label.item(), 1)
        self.assertTrue(torch.isfinite(sequence).all())
        self.assertAlmostEqual(sequence[0, 0, 0].item(), 0.1)
        self.assertEqual(sequence[-1, 0, 2].item(), 0.0)

    def test_train_and_test_evaluation_use_separate_manifest_splits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pose_root = root / "poses"
            pose_root.mkdir()
            records = []
            for index, split in enumerate(
                ("train", "train", "validation", "validation", "test", "test")
            ):
                label = "bad" if index % 2 == 0 else "good"
                keypoints = np.full((3, 2, 4), 0.25, dtype=np.float32)
                keypoints[..., 3] = 0.9
                pose_name = f"pose_{index}.npz"
                np.savez_compressed(pose_root / pose_name, keypoints=keypoints)
                records.append(
                    {
                        "video_path": f"video_{index}.mp4",
                        "pose_path": pose_name,
                        "label": label,
                        "split": split,
                    }
                )
            index = pd.DataFrame.from_records(records)
            for model_name in ("fc", "cnn_lstm"):
                checkpoint = root / f"{model_name}.pt"
                history = train_pose_classifier(
                    index,
                    pose_root,
                    model_name=model_name,
                    output_path=checkpoint,
                    sequence_length=4,
                    epochs=1,
                    batch_size=2,
                    seed=1,
                    device="cpu",
                )
                metrics = evaluate_pose_classifier(
                    index,
                    pose_root,
                    checkpoint,
                    split="test",
                    batch_size=2,
                    device="cpu",
                )
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(len(history), 1)
                self.assertEqual(metrics["support"], 2)
                self.assertEqual(metrics["split"], "test")
                self.assertEqual(metrics["class_order"], ["bad", "good"])


if __name__ == "__main__":
    unittest.main()
