"""The model's input columns and two simple usage ratios."""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
NUMERIC = [
    "call_failure",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age",
]
CATEGORICAL = ["complains", "age_group", "tariff_plan", "status"]
INPUT_COLUMNS = NUMERIC + CATEGORICAL
DERIVED = ["seconds_per_call", "call_failure_rate"]


class Customer(BaseModel):
    """One valid row; preparation and scoring share these input rules."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    customer_id: str = Field(min_length=1)
    call_failure: int = Field(ge=0)
    subscription_length: int = Field(ge=0)
    charge_amount: int = Field(
        ge=0, le=10
    )  # CSV includes 10 despite UCI's 0-9 description.
    seconds_of_use: int = Field(ge=0)
    frequency_of_use: int = Field(ge=0)
    frequency_of_sms: int = Field(ge=0)
    distinct_called_numbers: int = Field(ge=0)
    age: int = Field(ge=0, le=120)
    complains: int = Field(ge=0, le=1)
    age_group: int = Field(ge=1, le=5)
    tariff_plan: int = Field(ge=1, le=2)
    status: int = Field(ge=1, le=2)
    customer_value: float = Field(ge=0)


def add_features(customers):
    """Recompute ratios inside the saved pipeline, even if supplied in a CSV."""
    frame = customers.copy()
    frame["seconds_per_call"] = frame["seconds_of_use"] / (
        frame["frequency_of_use"] + 1
    )
    frame["call_failure_rate"] = frame["call_failure"] / (
        frame["frequency_of_use"] + frame["call_failure"] + 1
    )
    return frame


def validate_customers(frame):
    if frame.empty:
        raise ValueError("The customer file is empty.")
    required = ["customer_id", *INPUT_COLUMNS, "customer_value"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    records = [
        Customer.model_validate(row).model_dump()
        for row in frame[required].to_dict(orient="records")
    ]
    result = pd.DataFrame(records)
    if result["customer_id"].duplicated().any():
        raise ValueError("customer_id must be unique within a batch.")
    return result
