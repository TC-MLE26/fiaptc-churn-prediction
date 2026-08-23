import pandas as pd
import pytest

from src.ml.preprocessing import (
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


def _raw_row(**overrides):
    row = {
        "CustomerID": "0000-AAAAA",
        "Count": 1,
        "Country": "United States",
        "State": "California",
        "City": "Los Angeles",
        "Zip Code": 90003,
        "Lat Long": "33.96, -118.27",
        "Latitude": 33.96,
        "Longitude": -118.27,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 12,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Mailed check",
        "Monthly Charges": 53.85,
        "Total Charges": "646.20",
        "Churn Label": "No",
        "Churn Value": 0,
        "Churn Score": 40,
        "CLTV": 3239,
        "Churn Reason": None,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def raw_dataframe():
    return pd.DataFrame(
        [
            _raw_row(
                CustomerID="3668-QPYBK",
                **{
                    "Tenure Months": 2,
                    "Churn Label": "Yes",
                    "Churn Value": 1,
                    "Churn Score": 86,
                    "Churn Reason": "Competitor made better offer",
                },
            ),
            _raw_row(
                CustomerID="9237-HQITU",
                **{
                    "Tenure Months": 24,
                    "Senior Citizen": "Yes",
                    "Contract": "Two year",
                    "Internet Service": "Fiber optic",
                    "Total Charges": "1696.80",
                },
            ),
            _raw_row(
                CustomerID="0000-NEWBI",
                **{
                    "Tenure Months": 0,
                    "Phone Service": "No",
                    "Multiple Lines": "No phone service",
                    "Total Charges": " ",
                },
            ),
        ]
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
    assert normalize_column_name(source) == expected


def test_normalize_columns_renames_every_column(raw_dataframe):
    result = normalize_columns(raw_dataframe)

    assert all(column == column.lower() for column in result.columns)
    assert not any(" " in column for column in result.columns)
    assert "tenure_months" in result.columns


def test_add_target_from_churn_value(raw_dataframe):
    result = add_target(normalize_columns(raw_dataframe))

    assert result[TARGET].tolist() == [1, 0, 0]


def test_add_target_falls_back_to_churn_label():
    frame = pd.DataFrame({"churn_label": ["Yes", "no", " YES "]})

    assert add_target(frame)[TARGET].tolist() == [1, 0, 1]


def test_add_target_without_churn_column_raises():
    with pytest.raises(KeyError, match="No churn column found"):
        add_target(pd.DataFrame({"tenure_months": [1]}))


def test_coerce_total_charges_fills_blank_with_zero():
    frame = pd.DataFrame({"total_charges": ["108.15", " ", "", "1696.80"]})

    result = coerce_total_charges(frame)

    assert result["total_charges"].tolist() == [108.15, 0.0, 0.0, 1696.80]
    assert pd.api.types.is_numeric_dtype(result["total_charges"])


def test_drop_non_predictive_columns_removes_leakage(raw_dataframe):
    result = drop_non_predictive_columns(normalize_columns(raw_dataframe))

    for column in LEAKAGE_COLUMNS:
        assert column not in result.columns


def test_clean_data_keeps_only_the_expected_features(raw_dataframe):
    result = clean_data(raw_dataframe)

    assert list(result.columns) == FEATURES + [TARGET]
    assert len(FEATURES) == 19


def test_clean_data_removes_leakage_and_identifiers(raw_dataframe):
    result = clean_data(raw_dataframe)

    for column in ("churn_score", "churn_reason", "cltv", "churn_label", "churn_value"):
        assert column not in result.columns
    for column in ("customer_id", "city", "zip_code", "lat_long", "latitude"):
        assert column not in result.columns


def test_clean_data_has_no_nulls_and_string_categoricals(raw_dataframe):
    result = clean_data(raw_dataframe)

    assert not result.isna().to_numpy().any()
    for column in CATEGORICAL_FEATURES:
        assert result[column].dtype == object


def test_clean_data_converts_total_charges_of_new_customer(raw_dataframe):
    result = clean_data(raw_dataframe)
    new_customer = result[result["tenure_months"] == 0]

    assert new_customer["total_charges"].tolist() == [0.0]


def test_clean_data_missing_feature_raises(raw_dataframe):
    incomplete = raw_dataframe.drop(columns=["Contract"])

    with pytest.raises(ValueError, match="Missing expected features"):
        clean_data(incomplete)


def test_clean_data_does_not_mutate_the_input(raw_dataframe):
    before = raw_dataframe.copy()

    clean_data(raw_dataframe)

    pd.testing.assert_frame_equal(raw_dataframe, before)


def test_split_features_target(raw_dataframe):
    features, target = split_features_target(clean_data(raw_dataframe))

    assert list(features.columns) == FEATURES
    assert target.name == TARGET
    assert len(features) == len(target) == 3
