"""Evaluate a saved pose classifier on a manifest split."""

import argparse
import json
from pathlib import Path

from .training import evaluate_pose_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Pose manifest CSV.")
    parser.add_argument(
        "--pose-root",
        type=Path,
        required=True,
        help="Root directory used to resolve relative pose_path values.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metrics = evaluate_pose_classifier(
        index=args.index,
        pose_root=args.pose_root,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    output_path = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_{args.split}_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
