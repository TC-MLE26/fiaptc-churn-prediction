"""Cleaning rules for the Telco Customer Churn (IBM) dataset."""

from __future__ import annotations
import re
import pandas as pd

TARGET = "churn"

COLUMN_ALIASES = {
    "customerid": "customer_id",
    "seniorcitizen": "senior_citizen",
    "tenure": "tenure_months",
    "monthlycharges": "monthly_charges",
    "totalcharges": "total_charges",
    "churnvalue": "churn_value",
    "churnlabel": "churn_label",
}

LEAKAGE_COLUMNS = [
    "churn_label",
    "churn_value",
    "churn_score",
    "churn_reason",
    "cltv",
]

ID_AND_GEO_COLUMNS = [
    "customer_id",
    "count",
    "country",
    "state",
    "city",
    "zip_code",
    "lat_long",
    "latitude",
    "longitude",
]

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

UNBILLED_TOTAL_CHARGES = 0.0


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return COLUMN_ALIASES.get(normalized, normalized)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "churn_value" in result.columns:
        result[TARGET] = result["churn_value"].astype(int)
    elif "churn_label" in result.columns:
        label = result["churn_label"].astype(str).str.strip().str.lower()
        result[TARGET] = (label == "yes").astype(int)
    else:
        raise KeyError(
            "No churn column found. Expected 'churn_value' or 'churn_label', "
            f"got: {list(result.columns)}"
        )

    return result


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "total_charges" in result.columns:
        result["total_charges"] = pd.to_numeric(
            result["total_charges"], errors="coerce"
        ).fillna(UNBILLED_TOTAL_CHARGES)

    return result


def drop_non_predictive_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        column
        for column in LEAKAGE_COLUMNS + ID_AND_GEO_COLUMNS
        if column in df.columns
    ]
    return df.drop(columns=to_drop)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    result = drop_non_predictive_columns(
        coerce_total_charges(add_target(normalize_columns(df)))
    )

    missing = [column for column in FEATURES if column not in result.columns]
    if missing:
        raise ValueError(f"Missing expected features after cleaning: {missing}")

    result[CATEGORICAL_FEATURES] = result[CATEGORICAL_FEATURES].astype(str)
    result = result[FEATURES + [TARGET]]

    if result.isna().to_numpy().any():
        null_columns = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Null values remain after cleaning: {null_columns}")

    return result


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]
