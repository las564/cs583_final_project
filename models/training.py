"""Training and evaluation helpers for pose-sequence classifiers."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .classifiers import build_pose_classifier
from .dataset import INPUT_CHANNELS, LABEL_TO_ID, PoseSequenceDataset, load_pose_index


def select_device(device: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda/mps to an available PyTorch device."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = frame.loc[frame["split"].eq(split)].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"Pose index has no records for the {split!r} split.")
    return selected


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_records = 0

    with torch.set_grad_enabled(is_training):
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(sequences)
            loss = loss_function(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_records += batch_size

    return total_loss / total_records, total_correct / total_records


def train_pose_classifier(
    index: pd.DataFrame | str | Path,
    pose_root: str | Path,
    model_name: str,
    output_path: str | Path,
    sequence_length: int = 64,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "auto",
) -> list[dict[str, float]]:
    """Train on train rows and select the checkpoint with lowest validation loss."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")

    frame = load_pose_index(index)
    train_records = _validate_split(frame, "train")
    validation_records = _validate_split(frame, "validation")
    if set(train_records["label"]) != set(LABEL_TO_ID):
        raise ValueError("The training split must contain both good and bad labels.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = select_device(device)

    train_dataset = PoseSequenceDataset(train_records, pose_root, sequence_length)
    validation_dataset = PoseSequenceDataset(
        validation_records, pose_root, sequence_length
    )
    if validation_dataset.num_keypoints != train_dataset.num_keypoints:
        raise ValueError("Train and validation poses have different keypoint counts.")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    model = build_pose_classifier(
        model_name,
        num_keypoints=train_dataset.num_keypoints,
        sequence_length=sequence_length,
        input_channels=INPUT_CHANNELS,
    ).to(selected_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_validation_loss = float("inf")
    history = []
    checkpoint_path = Path(output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, loss_function, selected_device, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, loss_function, selected_device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_name": model_name,
                    "num_keypoints": train_dataset.num_keypoints,
                    "sequence_length": sequence_length,
                    "input_channels": INPUT_CHANNELS,
                    "class_to_id": LABEL_TO_ID,
                    "model_state_dict": model.state_dict(),
                    "best_epoch": epoch,
                    "best_validation_loss": best_validation_loss,
                    "seed": seed,
                },
                checkpoint_path,
            )

    return history


def _classification_metrics(
    labels: list[int], predictions: list[int]
) -> dict[str, Any]:
    true_positive = sum(
        truth == 1 and prediction == 1
        for truth, prediction in zip(labels, predictions)
    )
    true_negative = sum(
        truth == 0 and prediction == 0
        for truth, prediction in zip(labels, predictions)
    )
    false_positive = sum(
        truth == 0 and prediction == 1
        for truth, prediction in zip(labels, predictions)
    )
    false_negative = sum(
        truth == 1 and prediction == 0
        for truth, prediction in zip(labels, predictions)
    )
    total = len(labels)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (true_positive + true_negative) / total,
        "precision_good": precision,
        "recall_good": recall,
        "f1_good": f1,
        "support": total,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "class_order": ["bad", "good"],
    }


def evaluate_pose_classifier(
    index: pd.DataFrame | str | Path,
    pose_root: str | Path,
    checkpoint_path: str | Path,
    split: str = "test",
    batch_size: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate a saved checkpoint on one manifest split."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    frame = load_pose_index(index)
    records = _validate_split(frame, split.lower())
    selected_device = select_device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=True,
    )

    dataset = PoseSequenceDataset(
        records,
        pose_root,
        sequence_length=int(checkpoint["sequence_length"]),
    )
    if dataset.num_keypoints != int(checkpoint["num_keypoints"]):
        raise ValueError("Evaluation poses do not match the checkpoint keypoint count.")
    model = build_pose_classifier(
        str(checkpoint["model_name"]),
        num_keypoints=int(checkpoint["num_keypoints"]),
        sequence_length=int(checkpoint["sequence_length"]),
        input_channels=int(checkpoint["input_channels"]),
    ).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    actual_labels = []
    predicted_labels = []
    with torch.no_grad():
        for sequences, labels in loader:
            logits = model(sequences.to(selected_device))
            actual_labels.extend(labels.tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = _classification_metrics(actual_labels, predicted_labels)
    metrics["split"] = split.lower()
    metrics["model_name"] = checkpoint["model_name"]
    return metrics
