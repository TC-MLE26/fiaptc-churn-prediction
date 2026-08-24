"""Candidate model pipelines for the churn classification task."""

from __future__ import annotations
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.ml.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEED = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def build_candidates(seed: int = SEED) -> dict[str, Pipeline]:
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "logistic_regression_balanced": build_pipeline(
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            )
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "mlp": build_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=seed,
            )
        ),
    }
