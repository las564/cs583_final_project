"""Baseline fully connected and temporal CNN-LSTM classifiers."""

import torch
from torch import nn


class FullyConnectedPoseClassifier(nn.Module):
    """Classify a fixed-length pose sequence using a dense baseline."""

    def __init__(
        self,
        num_keypoints: int,
        sequence_length: int = 64,
        input_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        flattened_size = num_keypoints * sequence_length * input_channels
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        expected_shape = (
            self.sequence_length,
            self.num_keypoints,
            self.input_channels,
        )
        if sequence.ndim != 4 or tuple(sequence.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected input shape (batch, {expected_shape}), "
                f"received {tuple(sequence.shape)}."
            )
        return self.network(sequence)


class CNNLSTMPoseClassifier(nn.Module):
    """Apply temporal convolutions followed by an LSTM over frame features."""

    def __init__(
        self,
        num_keypoints: int,
        sequence_length: int = 64,
        input_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        frame_features = num_keypoints * input_channels
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(frame_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        expected_shape = (
            self.sequence_length,
            self.num_keypoints,
            self.input_channels,
        )
        if sequence.ndim != 4 or tuple(sequence.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected input shape (batch, {expected_shape}), "
                f"received {tuple(sequence.shape)}."
            )

        batch_size = sequence.shape[0]
        frame_features = sequence.reshape(batch_size, self.sequence_length, -1)
        temporal_features = self.temporal_cnn(frame_features.transpose(1, 2))
        recurrent_features, _ = self.lstm(temporal_features.transpose(1, 2))
        return self.classifier(recurrent_features[:, -1])


def build_pose_classifier(
    model_name: str,
    num_keypoints: int,
    sequence_length: int,
    input_channels: int = 3,
) -> nn.Module:
    """Construct a classifier by its command-line/API name."""
    model_types = {
        "fc": FullyConnectedPoseClassifier,
        "cnn_lstm": CNNLSTMPoseClassifier,
    }
    try:
        model_type = model_types[model_name]
    except KeyError:
        raise ValueError(
            f"Unknown model {model_name!r}; choose 'fc' or 'cnn_lstm'."
        ) from None
    return model_type(
        num_keypoints=num_keypoints,
        sequence_length=sequence_length,
        input_channels=input_channels,
    )
