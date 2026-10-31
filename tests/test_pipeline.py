"""
Automated Verification Suite for Statement #31 — Predictive Underwriting (2026 Update)
Run with:
    python -m unittest discover -s tests
"""

import unittest
from app import app


class TestPredictiveUnderwritingApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_page_loads(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Predictive Underwriting", res.data)

    def test_dashboard_summary_endpoint(self):
        res = self.client.get("/api/dashboard_summary")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("training_summary", data)
        self.assertTrue(
            data["training_summary"]["fairness_audit"]["after_mitigation"]["passes_80_percent_rule"]
        )

    def test_realtime_underwrite_endpoint(self):
        payload = {
            "applicant_name": "Test Applicant",
            "national_id": "ID-1234-5678",
            "region_income_tier": "Low-Income Zone",
            "driver_age": 31,
            "prior_claims_3yr": 0,
            "credit_stability_score": 710,
            "vehicle_value_usd": 25000,
            "vehicle_safety_rating": 5,
            "annual_mileage_km": 11000,
            "rt_telematics_score": 88.0,
            "rt_harsh_braking_per_100km": 1.1,
            "rt_night_driving_ratio": 0.08,
            "rt_overspeed_ratio": 0.05,
            "rt_weather_route_hazard_idx": 0.25,
            "enable_fairness": True,
            "enable_privacy": True,
            "privacy_epsilon": 2.5,
        }
        res = self.client.post("/api/underwrite", json=payload)
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertGreater(body["result"]["pricing"]["annual_technical_premium_usd"], 200.0)
        self.assertTrue(body["result"]["privacy_audit"]["applicant_token"].startswith("TOK-"))

    def test_simulate_stream_and_drift_endpoints(self):
        res_stream = self.client.get("/api/simulate_stream")
        self.assertEqual(res_stream.status_code, 200)
        self.assertIn("quote", res_stream.get_json())

        res_drift = self.client.get("/api/drift_scenario/q3_severe_drift")
        self.assertEqual(res_drift.status_code, 200)
        self.assertGreater(len(res_drift.get_json()["feature_drift"]), 5)


if __name__ == "__main__":
    unittest.main()
