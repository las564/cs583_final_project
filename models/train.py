"""Train a fully connected or CNN-LSTM pose classifier."""

import argparse
from pathlib import Path

from .training import train_pose_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Pose manifest CSV.")
    parser.add_argument(
        "--pose-root",
        type=Path,
        required=True,
        help="Root directory used to resolve relative pose_path values.",
    )
    parser.add_argument(
        "--model",
        choices=("fc", "cnn_lstm"),
        required=True,
        help="Classifier architecture.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args()

    history = train_pose_classifier(
        index=args.index,
        pose_root=args.pose_root,
        model_name=args.model,
        output_path=args.output,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    best_epoch = min(history, key=lambda row: row["validation_loss"])
    print(
        f"Best validation loss {best_epoch['validation_loss']:.4f} "
        f"at epoch {int(best_epoch['epoch'])}; checkpoint: {args.output}"
    )


if __name__ == "__main__":
    main()
