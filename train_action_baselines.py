from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from action_training_common import (
    DEFAULT_DATASET_DIRNAME,
    DEFAULT_OUTPUT_DIR,
    compute_binary_metrics,
    compute_video_exact_match,
    ensure_dir,
    load_manifest_feature_columns,
    write_json,
)

try:
    import xgboost as xgb  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    xgb = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

try:
    import lightgbm as lgb  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    lgb = None
    LIGHTGBM_IMPORT_ERROR = exc
else:
    LIGHTGBM_IMPORT_ERROR = None


@dataclass
class ConstantBinaryModel:
    positive_probability: float
    model_name: str = "constant"

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        _ = features
        return np.full(features.shape[0], self.positive_probability, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 5 separate tree-based baseline models from the action dataset manifest."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_DATASET_DIRNAME,
        help="Directory containing action_dataset_manifest.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to action_dataset_manifest.csv. Default: <dataset-dir>/action_dataset_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "training_baseline",
        help="Directory for baseline models and evaluation artifacts.",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "xgboost", "lightgbm"),
        default="auto",
        help="Preferred backend. Default: auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to the tree backend when applicable.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of boosting rounds. Default: 300",
    )
    return parser.parse_args()


def choose_backend(model_type: str) -> str:
    if model_type == "xgboost":
        if xgb is None:
            raise RuntimeError(f"xgboost is not installed: {XGBOOST_IMPORT_ERROR}")
        return "xgboost"
    if model_type == "lightgbm":
        if lgb is None:
            raise RuntimeError(f"lightgbm is not installed: {LIGHTGBM_IMPORT_ERROR}")
        return "lightgbm"
    if xgb is not None:
        return "xgboost"
    if lgb is not None:
        return "lightgbm"
    raise RuntimeError(
        "Khong tim thay xgboost hoac lightgbm. Hay cai it nhat mot backend training truoc khi chay baseline."
    )


def sanitize_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> np.ndarray:
    matrix = df.reindex(columns=feature_columns).to_numpy(dtype=np.float32, copy=True)
    for col_idx, feature_name in enumerate(feature_columns):
        fill_value = fill_values.get(feature_name, 0.0)
        column = matrix[:, col_idx]
        mask = ~np.isfinite(column)
        if np.any(mask):
            column[mask] = fill_value
            matrix[:, col_idx] = column
    return matrix.astype(np.float32)


def compute_fill_values(train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    for feature_name in feature_columns:
        values = pd.to_numeric(train_df[feature_name], errors="coerce").to_numpy(dtype=np.float32)
        valid = values[np.isfinite(values)]
        fill_values[feature_name] = float(np.median(valid)) if valid.size else 0.0
    return fill_values


def fit_model(
    backend: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_estimators: int,
) -> object:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    if positives == 0 or negatives == 0:
        return ConstantBinaryModel(positive_probability=float(np.mean(y_train)))

    if backend == "xgboost":
        assert xgb is not None
        scale_pos_weight = negatives / positives if positives > 0 else 1.0
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(x_train, y_train)
        return model

    assert backend == "lightgbm"
    assert lgb is not None
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        class_weight="balanced",
        verbose=-1,
    )
    model.fit(x_train, y_train)
    return model


def predict_probabilities(model: object, features: np.ndarray) -> np.ndarray:
    if isinstance(model, ConstantBinaryModel):
        return model.predict_proba(features)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        if isinstance(probs, list):
            probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 2:
            return np.asarray(probs[:, 1], dtype=np.float32)
        return np.asarray(probs, dtype=np.float32).reshape(-1)
    raise TypeError(f"Model does not expose predict_proba: {type(model)!r}")


def evaluate_split(
    model: object,
    split_df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if split_df.empty:
        empty_predictions = split_df.copy()
        empty_predictions["y_true"] = []
        empty_predictions["y_prob"] = []
        empty_predictions["y_pred"] = []
        return empty_predictions, compute_binary_metrics(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32))

    features = sanitize_features(split_df, feature_columns, fill_values)
    y_true = split_df["target_label"].astype(int).to_numpy(dtype=np.int8)
    y_prob = predict_probabilities(model, features)
    y_pred = (y_prob >= 0.5).astype(np.int8)

    predictions = split_df[
        [
            "sample_id",
            "video_id",
            "action_id",
            "action_name",
            "segment_status",
            "split",
        ]
    ].copy()
    predictions["y_true"] = y_true
    predictions["y_prob"] = y_prob.astype(np.float32)
    predictions["y_pred"] = y_pred
    return predictions, compute_binary_metrics(y_true, y_prob)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else (dataset_dir / "action_dataset_manifest.csv").resolve()
    output_dir = args.output_dir.resolve()
    model_dir = ensure_dir(output_dir / "models")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    backend = choose_backend(args.model_type)
    manifest_df = pd.read_csv(manifest_path)
    usable_df = manifest_df[
        (manifest_df["is_labeled"] == 1)
        & (manifest_df["sequence_ready"] == 1)
        & (manifest_df["split"].isin(["train", "val", "test"]))
    ].copy()
    if usable_df.empty:
        raise SystemExit("Khong co sample nao co label + bounds de train baseline.")

    feature_columns = load_manifest_feature_columns(usable_df)
    if not feature_columns:
        raise SystemExit("Manifest khong co cot feature nao bat dau bang 'feat_'.")

    print(f"Train baseline backend: {backend}")
    print(f"Manifest: {manifest_path}")
    print(f"Usable rows: {len(usable_df)}")

    metrics_payload: dict[str, object] = {
        "backend": backend,
        "manifest_path": str(manifest_path),
        "feature_count": len(feature_columns),
        "actions": {},
        "video_exact_match": {},
    }
    all_predictions: list[pd.DataFrame] = []

    for action_id in range(1, 6):
        action_df = usable_df[usable_df["action_id"] == action_id].copy()
        train_df = action_df[action_df["split"] == "train"].copy()
        val_df = action_df[action_df["split"] == "val"].copy()
        test_df = action_df[action_df["split"] == "test"].copy()

        if train_df.empty:
            raise SystemExit(f"Action {action_id} khong co sample train nao.")

        keep_columns = []
        for feature_name in feature_columns:
            values = pd.to_numeric(train_df[feature_name], errors="coerce").to_numpy(dtype=np.float32)
            if np.any(np.isfinite(values)):
                keep_columns.append(feature_name)
        if not keep_columns:
            raise SystemExit(f"Action {action_id} khong con feature nao hop le sau khi loc NaN.")

        fill_values = compute_fill_values(train_df, keep_columns)
        x_train = sanitize_features(train_df, keep_columns, fill_values)
        y_train = train_df["target_label"].astype(int).to_numpy(dtype=np.int8)
        model = fit_model(
            backend=backend,
            x_train=x_train,
            y_train=y_train,
            seed=args.seed + action_id,
            n_estimators=args.n_estimators,
        )

        model_path = model_dir / f"action_{action_id}_{backend}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(
                {
                    "backend": backend,
                    "action_id": action_id,
                    "feature_columns": keep_columns,
                    "fill_values": fill_values,
                    "model": model,
                },
                handle,
            )

        action_metrics: dict[str, object] = {
            "model_path": str(model_path),
            "feature_count": len(keep_columns),
            "positive_train_count": int(np.sum(y_train == 1)),
            "negative_train_count": int(np.sum(y_train == 0)),
            "splits": {},
        }

        for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            predictions_df, split_metrics = evaluate_split(
                model=model,
                split_df=split_df,
                feature_columns=keep_columns,
                fill_values=fill_values,
            )
            predictions_df["model_backend"] = backend
            all_predictions.append(predictions_df)
            action_metrics["splits"][split_name] = split_metrics

        metrics_payload["actions"][f"action_{action_id}"] = action_metrics

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    predictions_path = output_dir / "baseline_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    video_exact_match_rows: list[dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        exact_match = compute_video_exact_match(predictions_df, split_name)
        metrics_payload["video_exact_match"][split_name] = {
            key: value for key, value in exact_match.items() if key != "rows"
        }
        video_exact_match_rows.extend(exact_match["rows"])

    video_exact_match_path = output_dir / "baseline_video_exact_match.csv"
    pd.DataFrame(video_exact_match_rows).to_csv(video_exact_match_path, index=False)

    metrics_path = output_dir / "baseline_metrics.json"
    write_json(metrics_path, metrics_payload)

    print(f"Models: {model_dir}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Exact match: {video_exact_match_path}")


if __name__ == "__main__":
    main()
