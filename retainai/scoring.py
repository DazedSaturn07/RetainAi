"""Load a locally trained model and rank a batch using explicit assumptions."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from retainai.features import OUTPUT, validate_customers


def load_model(path=None):
    path = Path(path) if path is not None else OUTPUT / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "Run notebooks 01_data.ipynb and 02_model.ipynb before scoring."
        )
    # Load only the artifact produced locally by this project.
    return joblib.load(path)


def score_customers(frame, bundle, capacity=250, cost=20.0, retention_rate=0.25):
    """Rank a supplied batch using an explicitly assumed retention scenario."""
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, (int, np.integer))
        or capacity < 0
    ):
        raise ValueError("Capacity must be a nonnegative whole number.")
    if (
        not np.isfinite(cost)
        or cost < 0
        or not np.isfinite(retention_rate)
        or not 0 <= retention_rate <= 1
    ):
        raise ValueError(
            "Cost must be finite/nonnegative and retention rate must be between 0 and 1."
        )
    customers = validate_customers(frame)
    probability = bundle["model"].predict_proba(customers)[:, 1]
    result = customers[["customer_id", "customer_value"]].copy()
    loss = probability * customers["customer_value"].to_numpy()
    benefit = loss * retention_rate - cost
    result["churn_probability"] = np.round(probability, 6)
    result["churn_prediction"] = (probability >= 0.5).astype(int)
    result["risk_tier"] = np.where(
        probability >= 0.7, "high", np.where(probability >= 0.3, "medium", "low")
    )
    result["expected_churn_loss"] = np.round(loss, 2)
    result["scenario_net_benefit"] = np.round(benefit, 2)
    # Rank and make contact decisions using full precision before rounding output.
    ranks = pd.Series(loss).rank(method="first", ascending=False).astype(int).to_numpy()
    result["priority_rank"] = ranks
    result["contact_recommended"] = (ranks <= capacity) & (benefit > 0)
    return result
