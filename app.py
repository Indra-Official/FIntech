"""
Flask Web Server & Live Demonstration Dashboard for Statement #31 — Predictive Underwriting (2026 Update)
Run with:
    python app.py
Opens at http://127.0.0.1:5000
"""

import os
import random
from flask import Flask, render_template, request, jsonify

from src.data_pipeline import (
    FEATURE_COLUMNS,
    generate_underwriting_dataset,
    ensure_datasets_saved,
)
from src.risk_models import PredictiveUnderwritingPipeline
from src.monitoring_engine import ModelMonitoringEngine
from src.privacy_engine import PrivacyGuardEngine

app = Flask(__name__)

# Initialize & Train Pipeline on Startup
print("[*] Initializing Statement #31 Predictive Underwriting Pipeline (2026 Update)...")
datasets_info = ensure_datasets_saved("data")
df_reference = datasets_info["df_ref"]
df_drift_q3 = datasets_info["df_drift"]

pipeline = PredictiveUnderwritingPipeline()
training_summary = pipeline.fit(df_reference)

monitoring_engine = ModelMonitoringEngine(n_bins=10)
privacy_engine = PrivacyGuardEngine()

# Precompute Monitoring & Privacy-Utility Curves
feature_drift_q3 = monitoring_engine.evaluate_feature_drift(df_reference, df_drift_q3, FEATURE_COLUMNS)
quarterly_monitoring = monitoring_engine.evaluate_quarterly_timeline(
    pipeline, generate_underwriting_dataset, FEATURE_COLUMNS
)


def _eval_dp_df(df_sanitized):
    probs = pipeline.predict_calibrated_prob(df_sanitized, apply_fairness=True)
    return pipeline.evaluate_predictions(df_sanitized["claim_occurred"].values, probs)


privacy_utility_curve = privacy_engine.evaluate_privacy_utility_curve(
    df_reference.sample(n=900, random_state=42).reset_index(drop=True),
    _eval_dp_df,
)
print("[+] Pipeline Ready! Serving Live Demonstration Dashboard.")


# Curated Demo Personas for Presentation Walkthrough
DEMO_PERSONAS = {
    "safe_commuter": {
        "label": "Demo Case 1: Safe Telematics Driver (Preferred Tier)",
        "applicant_name": "Ananya Sharma",
        "national_id": "ID-7482-3910",
        "gender": "Female",
        "region_income_tier": "Middle-Income Zone",
        "driver_age": 34,
        "years_licensed": 14,
        "prior_claims_3yr": 0,
        "credit_stability_score": 765,
        "vehicle_value_usd": 28000,
        "vehicle_safety_rating": 5,
        "annual_mileage_km": 10500,
        "policy_tenure_years": 6.5,
        "rt_telematics_score": 91.5,
        "rt_harsh_braking_per_100km": 0.8,
        "rt_night_driving_ratio": 0.06,
        "rt_overspeed_ratio": 0.04,
        "rt_weather_route_hazard_idx": 0.22,
    },
    "fairness_showcase": {
        "label": "Demo Case 2: Low-Income Zone Safe Driver (2026 Fairness Showcase)",
        "applicant_name": "Kabir Verma",
        "national_id": "ID-5920-8144",
        "gender": "Male",
        "region_income_tier": "Low-Income Zone",
        "driver_age": 29,
        "years_licensed": 9,
        "prior_claims_3yr": 0,
        "credit_stability_score": 635,
        "vehicle_value_usd": 19500,
        "vehicle_safety_rating": 4,
        "annual_mileage_km": 13000,
        "policy_tenure_years": 3.5,
        "rt_telematics_score": 84.0,
        "rt_harsh_braking_per_100km": 1.6,
        "rt_night_driving_ratio": 0.14,
        "rt_overspeed_ratio": 0.09,
        "rt_weather_route_hazard_idx": 0.35,
    },
    "realtime_speeder": {
        "label": "Demo Case 3: Clean History but Aggressive Real-Time Telemetry",
        "applicant_name": "Rohan Malhotra",
        "national_id": "ID-8812-6021",
        "gender": "Male",
        "region_income_tier": "High-Income Zone",
        "driver_age": 26,
        "years_licensed": 6,
        "prior_claims_3yr": 0,
        "credit_stability_score": 740,
        "vehicle_value_usd": 54000,
        "vehicle_safety_rating": 4,
        "annual_mileage_km": 22000,
        "policy_tenure_years": 1.5,
        "rt_telematics_score": 49.0,
        "rt_harsh_braking_per_100km": 7.8,
        "rt_night_driving_ratio": 0.46,
        "rt_overspeed_ratio": 0.42,
        "rt_weather_route_hazard_idx": 0.64,
    },
    "high_risk_decline": {
        "label": "Demo Case 4: Multi-Claim + Severe Real-Time Hazard (Referral/Decline)",
        "applicant_name": "Vikram Reddy",
        "national_id": "ID-3391-4409",
        "gender": "Male",
        "region_income_tier": "Middle-Income Zone",
        "driver_age": 22,
        "years_licensed": 3,
        "prior_claims_3yr": 3,
        "credit_stability_score": 540,
        "vehicle_value_usd": 42000,
        "vehicle_safety_rating": 2,
        "annual_mileage_km": 28500,
        "policy_tenure_years": 0.5,
        "rt_telematics_score": 34.5,
        "rt_harsh_braking_per_100km": 11.4,
        "rt_night_driving_ratio": 0.58,
        "rt_overspeed_ratio": 0.55,
        "rt_weather_route_hazard_idx": 0.78,
    },
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard_summary", methods=["GET"])
def get_dashboard_summary():
    """Returns full model calibration, fairness, privacy, and drift metrics for the UI."""
    return jsonify({
        "status": "ok",
        "training_summary": training_summary,
        "feature_drift_q3": feature_drift_q3,
        "quarterly_monitoring": quarterly_monitoring,
        "privacy_utility_curve": privacy_utility_curve,
        "demo_personas": DEMO_PERSONAS,
    })


@app.route("/api/underwrite", methods=["POST"])
def api_underwrite():
    """Executes real-time underwriting & actuarial pricing on submitted applicant features."""
    payload = request.get_json(force=True) or {}
    enable_fairness = bool(payload.get("enable_fairness", True))
    enable_privacy = bool(payload.get("enable_privacy", True))
    privacy_epsilon = float(payload.get("privacy_epsilon", 2.5))

    result = pipeline.underwrite_single_applicant(
        applicant_data=payload,
        enable_fairness=enable_fairness,
        enable_privacy=enable_privacy,
        privacy_epsilon=privacy_epsilon,
    )
    return jsonify({"status": "ok", "result": result})


@app.route("/api/simulate_stream", methods=["GET"])
def api_simulate_stream():
    """Generates a live random IoT telematics + applicant event for the live stream demo."""
    sample_df = generate_underwriting_dataset(n_samples=1, random_state=random.randint(1, 99999))
    row = sample_df.iloc[0].to_dict()
    quote = pipeline.underwrite_single_applicant(
        applicant_data=row,
        enable_fairness=True,
        enable_privacy=True,
        privacy_epsilon=2.5,
    )
    return jsonify({
        "status": "ok",
        "applicant": {
            "policy_id": f"POL-RT-{random.randint(200000, 999999)}",
            "applicant_name": row["applicant_name"],
            "national_id": row["national_id"],
            "gender": row["gender"],
            "region_income_tier": row["region_income_tier"],
            "driver_age": int(row["driver_age"]),
            "prior_claims_3yr": int(row["prior_claims_3yr"]),
            "credit_stability_score": int(row["credit_stability_score"]),
            "vehicle_value_usd": float(row["vehicle_value_usd"]),
            "vehicle_safety_rating": int(row["vehicle_safety_rating"]),
            "annual_mileage_km": float(row["annual_mileage_km"]),
            "rt_telematics_score": float(row["rt_telematics_score"]),
            "rt_harsh_braking_per_100km": float(row["rt_harsh_braking_per_100km"]),
            "rt_night_driving_ratio": float(row["rt_night_driving_ratio"]),
            "rt_overspeed_ratio": float(row["rt_overspeed_ratio"]),
            "rt_weather_route_hazard_idx": float(row["rt_weather_route_hazard_idx"]),
        },
        "quote": quote,
    })


@app.route("/api/drift_scenario/<scenario>", methods=["GET"])
def api_drift_scenario(scenario: str):
    """Evaluates PSI and KS drift dynamically for any selected scenario in the UI."""
    valid = {"baseline", "q1_normal", "q2_mild_drift", "q3_severe_drift"}
    if scenario not in valid:
        scenario = "q3_severe_drift"
    df_cur = generate_underwriting_dataset(n_samples=1000, random_state=77, drift_scenario=scenario)
    drift_report = monitoring_engine.evaluate_feature_drift(df_reference, df_cur, FEATURE_COLUMNS)
    preds = pipeline.predict_calibrated_prob(df_cur)
    perf = pipeline.evaluate_predictions(df_cur["claim_occurred"].values, preds)
    return jsonify({
        "status": "ok",
        "scenario": scenario,
        "performance": perf,
        "feature_drift": drift_report,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
