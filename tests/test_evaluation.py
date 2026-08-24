import pandas as pd
import pytest

from src.ml.evaluation import (
    compare_models,
    cross_validate,
    evaluate,
    select_champion,
)
from src.ml.pipelines import build_pipeline
from src.ml.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from sklearn.linear_model import LogisticRegression

CATEGORY_BY_CHURN = {
    0: {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "Yes",
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Mailed check",
    },
    1: {
        "gender": "Female",
        "senior_citizen": "Yes",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
    },
}


def _clean_frame(rows=60):
    records = []
    for index in range(rows):
        churn = index % 2
        records.append(
            {
                "tenure_months": 2 + index if churn else 40 + index,
                "monthly_charges": 90.0 - index if churn else 30.0 + index,
                "total_charges": 100.0 + index if churn else 2000.0 + index,
                **CATEGORY_BY_CHURN[churn],
                TARGET: churn,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture()
def dataset():
    frame = _clean_frame()
    return frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], frame[TARGET]


@pytest.fixture()
def fitted_model(dataset):
    features, target = dataset
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(features, target)


def test_evaluate_returns_the_three_metrics(fitted_model, dataset):
    metrics = evaluate(fitted_model, *dataset)

    assert set(metrics) == {"f1", "recall", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_cross_validate_returns_mean_and_std(dataset):
    features, target = dataset
    scores = cross_validate(build_pipeline(LogisticRegression(max_iter=1000)), features, target)

    assert set(scores) == {"f1_cv_mean", "f1_cv_std"}
    assert 0.0 <= scores["f1_cv_mean"] <= 1.0
    assert scores["f1_cv_std"] >= 0.0


def test_cross_validate_is_reproducible(dataset):
    features, target = dataset
    model = build_pipeline(LogisticRegression(max_iter=1000))

    first = cross_validate(model, features, target, seed=1)
    second = cross_validate(model, features, target, seed=1)

    assert first == second


def test_compare_models_is_sorted_by_f1(fitted_model, dataset):
    comparison = compare_models({"a": fitted_model, "b": fitted_model}, *dataset)

    assert list(comparison.columns[:1]) == ["model"]
    assert comparison["f1"].is_monotonic_decreasing


def test_compare_models_includes_cross_validation_when_given(fitted_model, dataset):
    comparison = compare_models(
        {"a": fitted_model},
        *dataset,
        cross_validation={"a": {"f1_cv_mean": 0.5, "f1_cv_std": 0.1}},
    )

    assert comparison.loc[0, "f1_cv_mean"] == 0.5


def test_select_champion_picks_the_highest_f1():
    comparison = pd.DataFrame(
        [
            {"model": "worse", "f1": 0.40},
            {"model": "better", "f1": 0.80},
        ]
    )

    assert select_champion(comparison) == "better"
