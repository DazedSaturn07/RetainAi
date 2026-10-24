"""Small checks for shared rules; notebooks check splitting, fitting, and drift."""

import unittest

import numpy as np
import pandas as pd

from retainai.features import DATA, add_features, validate_customers
from retainai.scoring import score_customers


class FixedModel:
    """A known probability makes the expected arithmetic easy to verify."""

    def predict_proba(self, frame):
        return np.tile([0.2, 0.8], (len(frame), 1))


class HelperTests(unittest.TestCase):
    def setUp(self):
        # Three real example rows; no trained model or output folder required.
        self.customers = pd.read_csv(DATA / "customer_churn.csv").head(3)
        self.customers.columns = [
            "_".join(c.lower().split()) for c in self.customers.columns
        ]
        self.customers["customer_id"] = ["A", "B", "C"]
        self.customers["customer_value"] = [1000, 500, 10]
        self.bundle = {"model": FixedModel()}

    def test_zero_usage_has_finite_ratios(self):
        frame = self.customers.assign(
            seconds_of_use=0, frequency_of_use=0, call_failure=0
        )
        ratios = add_features(frame)[["seconds_per_call", "call_failure_rate"]]
        self.assertTrue((ratios == 0).all().all())

    def test_missing_invalid_and_duplicate_inputs_are_rejected(self):
        bad_frames = [
            self.customers.drop(columns="call_failure"),
            self.customers.assign(customer_id="same"),
            self.customers.assign(customer_value=float("inf")),
            self.customers.assign(complains=3),
            self.customers.assign(customer_id=""),
            self.customers.assign(call_failure=-1),
            self.customers.assign(age=121),
            self.customers.assign(charge_amount=11),
            self.customers.assign(tariff_plan=3),
            self.customers.assign(age_group=0),
            self.customers.assign(status=3),
            self.customers.assign(customer_value=float("nan")),
            self.customers.iloc[:0],
        ]
        for frame in bad_frames:
            with (
                self.subTest(columns=frame.columns.tolist()),
                self.assertRaises(ValueError),
            ):
                validate_customers(frame)

    def test_capacity_and_scenario_arithmetic(self):
        result = score_customers(self.customers, self.bundle, capacity=1)
        self.assertEqual(result.expected_churn_loss.tolist(), [800, 400, 8])
        self.assertEqual(result.scenario_net_benefit.tolist(), [180, 80, -18])
        self.assertEqual(result.contact_recommended.tolist(), [True, False, False])
        no_contacts = score_customers(self.customers, self.bundle, capacity=0)
        self.assertFalse(no_contacts.contact_recommended.any())

    def test_invalid_scenario_settings_are_rejected(self):
        for settings in [
            {"capacity": -1},
            {"capacity": 1.5},
            {"capacity": True},
            {"cost": float("nan")},
            {"cost": float("inf")},
            {"cost": -1},
            {"retention_rate": 2},
            {"retention_rate": -0.1},
            {"retention_rate": float("nan")},
        ]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                score_customers(self.customers, self.bundle, **settings)

    def test_equal_priorities_keep_input_order_and_negative_benefit_is_excluded(self):
        frame = self.customers.assign(customer_value=10)
        result = score_customers(frame, self.bundle)
        self.assertEqual(result.priority_rank.tolist(), [1, 2, 3])
        self.assertFalse(result.contact_recommended.any())

    def test_source_charge_tier_ten_is_preserved(self):
        frame = self.customers.assign(charge_amount=10)
        self.assertEqual(validate_customers(frame).charge_amount.tolist(), [10, 10, 10])

    def test_contact_decisions_use_unrounded_benefit(self):
        # All three benefits display as zero; their signs still control contact.
        frame = self.customers.assign(customer_value=[100.001, 99.999, 100.0])
        result = score_customers(frame, self.bundle, capacity=3)
        self.assertEqual(result.scenario_net_benefit.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(result.contact_recommended.tolist(), [True, False, False])

    def test_priority_uses_value_and_preserves_a_nondefault_index(self):
        class VaryingModel:
            def predict_proba(self, frame):
                return np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])

        frame = self.customers.assign(customer_value=[10, 1000, 100])
        frame.index = [10, 30, 20]
        result = score_customers(frame, {"model": VaryingModel()}, capacity=1)
        self.assertEqual(result.customer_id.tolist(), ["A", "B", "C"])
        self.assertEqual(result.priority_rank.tolist(), [3, 1, 2])
        self.assertEqual(result.contact_recommended.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
