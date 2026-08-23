"""Tests for src.services.preprocessing."""

import pandas as pd
import pytest

from src.services.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURES,
    LEAKAGE_COLUMNS,
    TARGET,
    add_target,
    clean_data,
    coerce_total_charges,
    drop_non_predictive_columns,
    normalize_column_name,
    normalize_columns,
    split_features_target,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Tenure Months", "tenure_months"),
        ("Monthly Charges", "monthly_charges"),
        ("Lat Long", "lat_long"),
        ("Zip Code", "zip_code"),
        ("CustomerID", "customer_id"),
        ("Churn Value", "churn_value"),
        ("tenure", "tenure_months"),
        ("TotalCharges", "total_charges"),
    ],
)
def test_normalize_column_name(source, expected):
    """Source column names become the snake_case names used downstream."""
    assert normalize_column_name(source) == expected


def test_normalize_columns_renames_every_column(raw_dataframe):
    """No column keeps a space or an uppercase letter after normalization."""
    result = normalize_columns(raw_dataframe)

    assert all(column == column.lower() for column in result.columns)
    assert not any(" " in column for column in result.columns)
    assert "tenure_months" in result.columns


def test_add_target_from_churn_value(raw_dataframe):
    """The numeric churn column becomes the 0/1 target."""
    result = add_target(normalize_columns(raw_dataframe))

    assert result[TARGET].tolist() == [1, 0, 0]


def test_add_target_falls_back_to_churn_label():
    """Without churn_value, the textual label is used."""
    frame = pd.DataFrame({"churn_label": ["Yes", "no", " YES "]})

    assert add_target(frame)[TARGET].tolist() == [1, 0, 1]


def test_add_target_without_churn_column_raises():
    """A dataset with no churn column fails loudly instead of silently."""
    with pytest.raises(KeyError, match="No churn column found"):
        add_target(pd.DataFrame({"tenure_months": [1]}))


def test_coerce_total_charges_fills_blank_with_zero():
    """Blank total_charges means the customer was never billed, so zero."""
    frame = pd.DataFrame({"total_charges": ["108.15", " ", "", "1696.80"]})

    result = coerce_total_charges(frame)

    assert result["total_charges"].tolist() == [108.15, 0.0, 0.0, 1696.80]
    assert pd.api.types.is_numeric_dtype(result["total_charges"])


def test_drop_non_predictive_columns_removes_leakage(raw_dataframe):
    """Columns that only exist after cancellation are dropped."""
    result = drop_non_predictive_columns(normalize_columns(raw_dataframe))

    for column in LEAKAGE_COLUMNS:
        assert column not in result.columns


def test_clean_data_keeps_only_the_expected_features(raw_dataframe):
    """The cleaned frame carries exactly the 19 features plus the target."""
    result = clean_data(raw_dataframe)

    assert list(result.columns) == FEATURES + [TARGET]
    assert len(FEATURES) == 19


def test_clean_data_removes_leakage_and_identifiers(raw_dataframe):
    """The most important guarantee: no leaked column reaches the model.

    churn_score, churn_reason and cltv are filled in by the operator after the
    customer cancels. If any of them stayed, the model would score almost
    perfectly offline and be worthless in production.
    """
    result = clean_data(raw_dataframe)

    for column in ("churn_score", "churn_reason", "cltv", "churn_label", "churn_value"):
        assert column not in result.columns
    for column in ("customer_id", "city", "zip_code", "lat_long", "latitude"):
        assert column not in result.columns


def test_clean_data_has_no_nulls_and_string_categoricals(raw_dataframe):
    """Cleaning leaves no nulls, and categoricals are strings.

    senior_citizen may arrive as Yes/No or as 1/0 depending on the file
    version; casting to string makes both behave the same for the encoder.
    """
    result = clean_data(raw_dataframe)

    assert not result.isna().to_numpy().any()
    for column in CATEGORICAL_FEATURES:
        assert result[column].dtype == object


def test_clean_data_converts_total_charges_of_new_customer(raw_dataframe):
    """The customer with zero months ends up with total_charges 0."""
    result = clean_data(raw_dataframe)
    new_customer = result[result["tenure_months"] == 0]

    assert new_customer["total_charges"].tolist() == [0.0]


def test_clean_data_missing_feature_raises(raw_dataframe):
    """A dataset without an expected feature fails with a clear message."""
    incomplete = raw_dataframe.drop(columns=["Contract"])

    with pytest.raises(ValueError, match="Missing expected features"):
        clean_data(incomplete)


def test_clean_data_does_not_mutate_the_input(raw_dataframe):
    """Cleaning returns a new frame and leaves the original untouched."""
    before = raw_dataframe.copy()

    clean_data(raw_dataframe)

    pd.testing.assert_frame_equal(raw_dataframe, before)


def test_split_features_target(raw_dataframe):
    """The split returns the feature matrix and the target separately."""
    features, target = split_features_target(clean_data(raw_dataframe))

    assert list(features.columns) == FEATURES
    assert target.name == TARGET
    assert len(features) == len(target) == 3
