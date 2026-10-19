"""Train and evaluate an arrival-delay regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Arrival Delay in Minutes"
NUMERIC_FEATURES = ["Age", "Flight Distance"]
DEPARTURE_DELAY_FEATURE = "Departure Delay in Minutes"
CATEGORICAL_FEATURES = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]
POST_DEPARTURE_FEATURES = [
    *NUMERIC_FEATURES,
    DEPARTURE_DELAY_FEATURE,
    *CATEGORICAL_FEATURES,
]
PRE_DEPARTURE_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an airline arrival-delay regressor."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data.csv"),
        help="Path to the airline CSV (default: data.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for the model and evaluation outputs.",
    )
    parser.add_argument(
        "--prediction-stage",
        choices=("after-departure", "pre-departure"),
        default="after-departure",
        help="Whether departure delay is known at prediction time.",
    )
    return parser.parse_args()


def read_data(path: Path, features: list[str]) -> tuple[pd.DataFrame, pd.Series, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = pd.read_csv(path)
    required = [*features, TARGET]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    rows_before_target_filter = len(data)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).copy()
    data = data.loc[data[TARGET] >= 0].copy()
    dropped_target_rows = rows_before_target_filter - len(data)
    if data.empty:
        raise ValueError(f"No valid non-negative values found for target {TARGET!r}.")

    numeric = [*NUMERIC_FEATURES, TARGET]
    if DEPARTURE_DELAY_FEATURE in features:
        numeric.append(DEPARTURE_DELAY_FEATURE)
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data[features], data[TARGET], dropped_target_rows


def build_pipeline(features: list[str]) -> Pipeline:
    numeric_features = [column for column in NUMERIC_FEATURES if column in features]
    if DEPARTURE_DELAY_FEATURE in features:
        numeric_features.append(DEPARTURE_DELAY_FEATURE)
    categorical_features = [
        column for column in CATEGORICAL_FEATURES if column in features
    ]
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", HistGradientBoostingRegressor(random_state=42)),
        ]
    )


def regression_metrics(
    actual: pd.Series | np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    return {
        "mae_minutes": float(mean_absolute_error(actual, predicted)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error_minutes": float(
            median_absolute_error(actual, predicted)
        ),
        "r2": float(r2_score(actual, predicted)),
        "within_5_minutes_pct": float(
            np.mean(np.abs(np.asarray(actual) - predicted) <= 5) * 100
        ),
        "within_15_minutes_pct": float(
            np.mean(np.abs(np.asarray(actual) - predicted) <= 15) * 100
        ),
        "mean_signed_error_minutes": float(np.mean(predicted - np.asarray(actual))),
    }


def delay_band_report(
    actual: pd.Series, predicted: np.ndarray
) -> pd.DataFrame:
    bands = pd.cut(
        actual,
        bins=[-np.inf, 0, 15, 60, np.inf],
        labels=["on-time (0 min)", "short (1-15 min)", "moderate (16-60 min)", "long (>60 min)"],
        right=True,
    )
    report: list[dict[str, Any]] = []
    for band in bands.cat.categories:
        mask = bands == band
        if not mask.any():
            continue
        band_actual = actual.loc[mask]
        band_predicted = predicted[mask.to_numpy()]
        report.append(
            {
                "delay_band": str(band),
                "count": int(mask.sum()),
                **regression_metrics(band_actual, band_predicted),
            }
        )
    return pd.DataFrame(report)


def main() -> None:
    args = parse_args()
    features = (
        PRE_DEPARTURE_FEATURES
        if args.prediction_stage == "pre-departure"
        else POST_DEPARTURE_FEATURES
    )
    x, y, dropped_target_rows = read_data(args.data, features)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(features)
    parameter_distributions = {
        "regressor__max_iter": [100, 200, 300],
        "regressor__learning_rate": [0.03, 0.06, 0.1],
        "regressor__max_leaf_nodes": [15, 31, 63],
        "regressor__l2_regularization": [0.0, 1.0, 5.0],
        "regressor__min_samples_leaf": [20, 50, 100],
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=8,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)

    model = search.best_estimator_
    test_predictions = np.maximum(0, model.predict(x_test))
    baseline_predictions = np.full(len(y_test), y_train.median(), dtype=float)
    metrics = {
        "dataset": str(args.data),
        "prediction_stage": args.prediction_stage,
        "target": TARGET,
        "features": features,
        "rows_used": int(len(x)),
        "rows_dropped_for_invalid_target": int(dropped_target_rows),
        "target_profile_minutes": {
            "mean": float(y.mean()),
            "median": float(y.median()),
            "p90": float(y.quantile(0.90)),
            "p95": float(y.quantile(0.95)),
            "p99": float(y.quantile(0.99)),
            "maximum": float(y.max()),
            "zero_delay_pct": float(y.eq(0).mean() * 100),
        },
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "baseline_median_predictor": regression_metrics(y_test, baseline_predictions),
        "tuned_hist_gradient_boosting": regression_metrics(y_test, test_predictions),
        "best_cv_mae_minutes": float(-search.best_score_),
        "best_parameters": search.best_params_,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "arrival_delay_model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    predictions = pd.DataFrame(
        {
            "actual_arrival_delay_minutes": y_test.to_numpy(),
            "predicted_arrival_delay_minutes": test_predictions,
            "signed_error_minutes": test_predictions - y_test.to_numpy(),
        },
        index=y_test.index,
    )
    predictions.to_csv(args.output_dir / "test_predictions.csv", index_label="row")
    delay_band_report(y_test, test_predictions).to_csv(
        args.output_dir / "delay_band_metrics.csv", index=False
    )

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": features,
            "mae_increase_minutes": importance.importances_mean,
            "std_dev_minutes": importance.importances_std,
        }
    ).sort_values("mae_increase_minutes", ascending=False)
    importance_frame.to_csv(
        args.output_dir / "permutation_importance.csv", index=False
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and reports to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
