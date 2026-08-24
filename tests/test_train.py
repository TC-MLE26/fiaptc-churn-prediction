import json

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.ml import train as train_module
from src.ml.train import save, train

RAW_BY_CHURN = {
    0: {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "Phone Service": "Yes",
        "Multiple Lines": "Yes",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection": "Yes",
        "Tech Support": "Yes",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Two year",
        "Paperless Billing": "No",
        "Payment Method": "Mailed check",
    },
    1: {
        "Gender": "Female",
        "Senior Citizen": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
    },
}


def _raw_frame(rows=80):
    records = []
    for index in range(rows):
        churn = index % 2
        records.append(
            {
                "CustomerID": f"{index:04d}-AAAAA",
                "Count": 1,
                "Country": "United States",
                "State": "California",
                "City": "Los Angeles",
                "Zip Code": 90003,
                "Lat Long": "33.96, -118.27",
                "Latitude": 33.96,
                "Longitude": -118.27,
                "Tenure Months": 2 + index if churn else 40 + index,
                "Monthly Charges": 90.0 - index * 0.5 if churn else 30.0 + index * 0.5,
                "Total Charges": str(100.0 + index) if churn else str(2000.0 + index),
                **RAW_BY_CHURN[churn],
                "Churn Label": "Yes" if churn else "No",
                "Churn Value": churn,
                "Churn Score": 90 if churn else 20,
                "CLTV": 3000,
                "Churn Reason": "Competitor" if churn else None,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture()
def trained(monkeypatch):
    monkeypatch.setattr(train_module, "load_raw", lambda *args, **kwargs: _raw_frame())
    return train()


def test_train_returns_model_name_and_comparison(trained):
    model, champion_name, comparison = trained

    assert isinstance(model, Pipeline)
    assert champion_name in set(comparison["model"])
    assert len(comparison) == 4


def test_train_champion_is_the_best_f1(trained):
    _, champion_name, comparison = trained

    assert champion_name == comparison.loc[comparison["f1"].idxmax(), "model"]


def test_train_returns_a_fitted_model(trained):
    model, _, _ = trained
    features = _raw_frame(2)

    from src.ml.preprocessing import clean_data, split_features_target

    sample, _ = split_features_target(clean_data(features))
    probabilities = model.predict_proba(sample)[:, 1]

    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_train_comparison_reports_every_metric(trained):
    _, _, comparison = trained

    for column in ("f1", "recall", "roc_auc", "f1_cv_mean", "f1_cv_std"):
        assert column in comparison.columns


def test_save_writes_the_model_and_the_metrics(trained, tmp_path, monkeypatch):
    model, champion_name, comparison = trained
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "CHAMPION_PATH", tmp_path / "champion_model.joblib")
    monkeypatch.setattr(train_module, "METRICS_PATH", tmp_path / "metrics.json")

    save(model, champion_name, comparison)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert (tmp_path / "champion_model.joblib").is_file()
    assert metrics["champion"] == champion_name
    assert len(metrics["comparison"]) == 4
    assert "sklearn_version" in metrics
