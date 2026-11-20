# cs583_final_project
Exercise Form Classification

## Dataset index

Install the CSV/indexing dependency:

```bash
python3 -m pip install -r requirements.txt
```

Create a CSV manifest of the raw videos before splitting or processing:

```bash
python3 scripts/create_dataset_index.py
```

By default, the script reads `data/raw` and writes
`data/metadata/dataset_index.csv`. Each CSV row contains a path relative to the
dataset root, exercise, form label, viewpoint, and subject ID. Paths can be
overridden:

```bash
python3 scripts/create_dataset_index.py \
  --dataset-root /path/to/data/raw \
  --output /path/to/dataset_index.csv
```

The CSV can be loaded directly into a dataframe, for example with pandas:

```python
import pandas as pd

dataset = pd.read_csv(
    "data/metadata/dataset_index.csv",
    dtype={"subject_id": "string"},
)
```

To create a participant-wise split for one exercise and viewpoint, pass both
filters. This writes only matching videos and adds a `split` column. The default
ratios are 70% train, 15% validation, and 15% test, assigned by participant;
set the seed and ratios for repeatable experiments:

```bash
python3 scripts/create_dataset_index.py \
  --exercise squat \
  --viewpoint front \
  --split-ratios 0.7 0.15 0.15 \
  --seed 42
```

All recordings for a participant are assigned to the same split. At least
three participants must be present in the selected exercise/viewpoint subset.
By default, the split CSV is written to
`data/metadata/squat_front_split_index.csv`. Pass `--output` to choose another
path.

## Pose extraction

Run pose extraction on a CSV index. Relative `video_path` values are resolved
against `--dataset-root`; the index may contain extra metadata such as labels
and participant splits, which are kept in the output manifest.

```bash
python3 -m pose_pipeline \
  --index data/metadata/squat_front_split_index.csv \
  --dataset-root data/raw \
  --backend yolo \
  --model yolo11n-pose.pt \
  --visualize
```

Ultralytics downloads the default weights on first use; pass `--model` with a
local weights path to use an existing model.

Use `--backend mediapipe` to swap estimators. Pose sequences are saved as
compressed NPZ files under `data/processed/poses/<backend>/`, and a
`pose_index.csv` manifest records each artifact path and video metadata. Each
NPZ contains `keypoints` with shape `(frames, keypoints, 4)` in `x, y, z,
confidence` order, plus frame indices, timestamps, and keypoint names. `x` and
`y` are normalized to image dimensions; depth and confidence use each backend's
native definitions. The backend tracking state is reset between videos. Use
`--frame-stride N` to process every Nth frame. For YOLO, the pipeline applies
an exponential moving average to visible keypoint x/y coordinates; the default
`--smoothing-alpha 0.4` reduces frame-to-frame jitter without filling missing
detections. Set it to `1` to disable smoothing (smaller values smooth more).
MediaPipe uses its built-in landmark smoothing. Both the saved poses and
visualizations use the smoothed sequence. Add `--visualize` to render each video
as part of the same run; by default the annotated videos go to
`data/processed/visualizations/<backend>/`, and the pose manifest records a
`visualization_path` for each output. Use `--visualization-dir` to change this
directory.

The pipeline API also accepts a pandas DataFrame directly, and takes the
estimator as an argument so other implementations can follow the same interface:

```python
import pandas as pd

from pose_pipeline import (
    MediaPipePoseEstimator,
    YOLOPoseEstimator,
    run_pose_estimation,
)

index = pd.read_csv(
    "data/metadata/squat_front_split_index.csv",
    dtype={"subject_id": "string"},
)
estimator = MediaPipePoseEstimator()  # Or YOLOPoseEstimator()
try:
    pose_index = run_pose_estimation(
        index,
        estimator,
        dataset_root="data/raw",
        output_dir="data/processed/poses/mediapipe",
    )
finally:
    estimator.close()
```

## Visualize poses

Render the saved keypoints over the original clip. The output is a new MP4;
source audio is not copied. Only frames present in the pose file are annotated
(with `--frame-stride 1`, every frame is annotated):

```bash
python -m pose_pipeline.visualize \
  --video data/raw/squat/good/front/subject_001_squat_good_front.mp4 \
  --pose data/processed/poses/yolo/squat/good/front/subject_001_squat_good_front.npz \
  --output data/processed/visualizations/subject_001_squat_good_front.mp4
```

Skeleton connections are drawn for the supported YOLO COCO-17 and MediaPipe
landmark names; other estimators still display keypoint dots.

## Train and evaluate pose classifiers

The classifier architectures, pose-sequence dataset, and train/evaluate
entrypoints live in the separate top-level `models/` package.

Install PyTorch, then train either the fully connected baseline (`fc`) or the
temporal CNN-LSTM (`cnn_lstm`) from the pose manifest:

```bash
python -m models.train \
  --index data/processed/poses/yolo/pose_index.csv \
  --pose-root data/processed/poses/yolo \
  --model fc \
  --output data/processed/models/squat_front_fc.pt
```

Change `--model fc` to `--model cnn_lstm` to train the CNN-LSTM. Both models
uniformly sample each pose sequence to 64 frames by default and use normalized
image `x`, `y`, and keypoint confidence as features. Missing coordinates are
filled with zero; depth is excluded because YOLO does not provide it. Training
uses only the `train` split and selects the checkpoint with the lowest
`validation` loss.

Evaluate the saved checkpoint on the held-out test participants:

```bash
python -m models.evaluate \
  --index data/processed/poses/yolo/pose_index.csv \
  --pose-root data/processed/poses/yolo \
  --checkpoint data/processed/models/squat_front_fc.pt
```

Evaluation defaults to `test` and writes accuracy, good-form precision/recall/F1,
and a confusion matrix alongside the checkpoint. The index's `split` column is
preserved through pose extraction and used to keep training, validation, and
test records separate.
