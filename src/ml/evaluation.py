"""Model evaluation and champion selection."""

from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score

CV_FOLDS = 5
SELECTION_METRIC = "f1"


def evaluate(model: BaseEstimator, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    from sklearn.metrics import f1_score, recall_score, roc_auc_score

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "f1": float(f1_score(target, predictions)),
        "recall": float(recall_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def cross_validate(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    folds: int = CV_FOLDS,
    seed: int = 42,
) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, features, target, cv=splitter, scoring=SELECTION_METRIC)

    return {"f1_cv_mean": float(scores.mean()), "f1_cv_std": float(scores.std())}


def compare_models(
    fitted_models: dict[str, BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    cross_validation: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows = []
    for name, model in fitted_models.items():
        row = {"model": name, **evaluate(model, features, target)}
        if cross_validation and name in cross_validation:
            row.update(cross_validation[name])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(SELECTION_METRIC, ascending=False).reset_index(drop=True)


def select_champion(comparison: pd.DataFrame) -> str:
    return str(comparison.loc[comparison[SELECTION_METRIC].idxmax(), "model"])
