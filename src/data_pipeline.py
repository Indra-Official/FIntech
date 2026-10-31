"""
Data Pipeline Module: Historical & Real-Time Insurance Underwriting Data
Generates and processes realistic insurance underwriting datasets combining:
1. Historical Data: Policy tenure, prior claims, credit stability score, asset/vehicle value, age, coverage history.
2. Real-Time Telemetry Data: Telematics driving score, harsh braking events/100km, night-driving ratio,
   speeding frequency, and real-time weather/route risk index.
3. Protected Attributes (for 2026 Fairness Auditing): Gender, Age Group, Socioeconomic Region Tier.
"""

import os
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # Historical Features
    "driver_age",
    "years_licensed",
    "prior_claims_3yr",
    "credit_stability_score",
    "vehicle_value_usd",
    "vehicle_safety_rating",
    "annual_mileage_km",
    "policy_tenure_years",
    # Real-Time Telematics & Contextual Stream Features
    "rt_telematics_score",
    "rt_harsh_braking_per_100km",
    "rt_night_driving_ratio",
    "rt_overspeed_ratio",
    "rt_weather_route_hazard_idx",
]

PROTECTED_COLUMNS = [
    "gender",              # 'Female', 'Male'
    "age_bracket",         # 'Young (<27)', 'Core (27-60)', 'Senior (>60)'
    "region_income_tier",  # 'Low-Income Zone', 'Middle-Income Zone', 'High-Income Zone'
]


def generate_underwriting_dataset(n_samples: int = 3500, random_state: int = 42, drift_scenario: str = "baseline") -> pd.DataFrame:
    """
    Generates a synthetic yet actuarially grounded insurance underwriting dataset
    combining historical records and real-time telematics streams.

    Parameters
    ----------
    n_samples : int
        Number of policies to generate.
    random_state : int
        Seed for reproducibility.
    drift_scenario : str
        'baseline', 'q1_normal', 'q2_mild_drift', 'q3_severe_drift'
        Used to simulate covariate and concept drift for 2026 Model Monitoring.
    """
    rng = np.random.default_rng(random_state)

    # Policy IDs and PII (for privacy demonstration)
    policy_ids = [f"POL-{100000 + i}" for i in range(n_samples)]
    first_names = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Kabir", "Meera", "Arjun", "Divya", "Rahul", "Neha"]
    last_names = ["Sharma", "Patel", "Verma", "Nair", "Iyer", "Gupta", "Reddy", "Singh", "Joshi", "Malhotra"]
    full_names = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n_samples)]
    national_ids = [f"ID-{rng.integers(1000, 9999)}-{rng.integers(1000, 9999)}" for _ in range(n_samples)]

    # Protected Demographic Attributes
    gender = rng.choice(["Female", "Male"], size=n_samples, p=[0.48, 0.52])
    region_income_tier = rng.choice(
        ["Low-Income Zone", "Middle-Income Zone", "High-Income Zone"],
        size=n_samples,
        p=[0.30, 0.45, 0.25],
    )

    # Historical Features
    driver_age = np.clip(rng.normal(loc=39, scale=13, size=n_samples).round(), 18, 78).astype(int)
    age_bracket = np.where(
        driver_age < 27,
        "Young (<27)",
        np.where(driver_age > 60, "Senior (>60)", "Core (27-60)"),
    )
    years_licensed = np.clip(driver_age - 18 - rng.integers(0, 4, size=n_samples), 0, 55)

    # Introduce historical proxy bias in credit score correlated with region_income_tier
    # (This makes the 2026 Fairness Constraint module meaningfully demonstrate bias reduction!)
    tier_offset = np.where(
        region_income_tier == "Low-Income Zone", -55,
        np.where(region_income_tier == "High-Income Zone", 40, 0)
    )
    credit_stability_score = np.clip(
        rng.normal(loc=690, scale=75, size=n_samples) + tier_offset,
        350, 850
    ).round().astype(int)

    prior_claims_3yr = rng.poisson(lam=0.38, size=n_samples)
    prior_claims_3yr = np.clip(prior_claims_3yr, 0, 5)

    vehicle_value_usd = np.clip(rng.lognormal(mean=10.2, sigma=0.45, size=n_samples), 8000, 120000).round(0)
    vehicle_safety_rating = rng.choice([2, 3, 4, 5], size=n_samples, p=[0.08, 0.27, 0.42, 0.23])
    annual_mileage_km = np.clip(rng.normal(loc=14500, scale=4800, size=n_samples), 3000, 45000).round(0)
    policy_tenure_years = np.clip(rng.exponential(scale=3.8, size=n_samples).round(1), 0.1, 20.0)

    # Real-Time Telematics & IoT Sensor Features
    rt_harsh_braking_per_100km = np.clip(rng.gamma(shape=2.1, scale=1.4, size=n_samples), 0.1, 18.0).round(2)
    rt_night_driving_ratio = np.clip(rng.beta(a=1.8, b=6.5, size=n_samples), 0.0, 0.85).round(3)
    rt_overspeed_ratio = np.clip(rng.beta(a=1.6, b=7.0, size=n_samples), 0.0, 0.75).round(3)
    rt_weather_route_hazard_idx = np.clip(rng.normal(loc=0.42, scale=0.18, size=n_samples), 0.05, 0.98).round(3)

    # Apply Drift Scenarios for Model Monitoring (2026 requirement)
    if drift_scenario == "q1_normal":
        rt_weather_route_hazard_idx = np.clip(rt_weather_route_hazard_idx + 0.02, 0.05, 0.99)
    elif drift_scenario == "q2_mild_drift":
        rt_harsh_braking_per_100km = np.clip(rt_harsh_braking_per_100km * 1.18 + 0.6, 0.1, 22.0)
        rt_night_driving_ratio = np.clip(rt_night_driving_ratio * 1.15 + 0.03, 0.0, 0.92)
        annual_mileage_km = np.clip(annual_mileage_km * 1.10, 3000, 50000)
    elif drift_scenario == "q3_severe_drift":
        rt_harsh_braking_per_100km = np.clip(rt_harsh_braking_per_100km * 1.48 + 1.8, 0.1, 25.0)
        rt_overspeed_ratio = np.clip(rt_overspeed_ratio * 1.40 + 0.08, 0.0, 0.95)
        rt_weather_route_hazard_idx = np.clip(rt_weather_route_hazard_idx + 0.19, 0.05, 0.99)
        credit_stability_score = np.clip(credit_stability_score - 45, 350, 850)

    # Composite Real-Time Telematics Score (0 to 100, higher is safer)
    raw_telematics = (
        95.0
        - 3.8 * rt_harsh_braking_per_100km
        - 32.0 * rt_overspeed_ratio
        - 22.0 * rt_night_driving_ratio
        - 12.0 * rt_weather_route_hazard_idx
        + rng.normal(0, 2.5, size=n_samples)
    )
    rt_telematics_score = np.clip(raw_telematics, 20.0, 99.0).round(1)

    # True Latent Risk Log-Odds (Actuarial Ground Truth combining Historical + Real-Time)
    logit = (
        -2.45
        + 0.52 * prior_claims_3yr
        + 0.24 * rt_harsh_braking_per_100km
        + 2.35 * rt_overspeed_ratio
        + 1.65 * rt_night_driving_ratio
        + 1.15 * rt_weather_route_hazard_idx
        - 0.025 * (rt_telematics_score - 70.0)
        + 0.000035 * (annual_mileage_km - 14000)
        - 0.16 * (vehicle_safety_rating - 3.5)
        - 0.06 * np.minimum(policy_tenure_years, 8.0)
        + np.where(driver_age < 25, 0.48, np.where(driver_age > 67, 0.28, 0.0))
        - 0.0022 * (credit_stability_score - 680)
        # Historical systemic penalty on Low-Income Zone (to test fairness mitigation)
        + np.where(region_income_tier == "Low-Income Zone", 0.38, 0.0)
    )

    true_claim_prob = 1.0 / (1.0 + np.exp(-logit))
    claim_occurred = rng.binomial(n=1, p=true_claim_prob, size=n_samples)

    # Claim Severity (Loss Amount in USD if a claim occurs)
    base_severity = (
        0.085 * vehicle_value_usd
        * (1.0 + 0.45 * rt_overspeed_ratio + 0.35 * rt_weather_route_hazard_idx)
        * (1.0 - 0.06 * (vehicle_safety_rating - 3))
    )
    log_noise = rng.lognormal(mean=0.0, sigma=0.38, size=n_samples)
    expected_severity_if_claim = np.clip(base_severity * log_noise, 600.0, 85000.0).round(2)
    observed_claim_loss = np.where(claim_occurred == 1, expected_severity_if_claim, 0.0).round(2)

    df = pd.DataFrame({
        "policy_id": policy_ids,
        "applicant_name": full_names,
        "national_id": national_ids,
        "gender": gender,
        "age_bracket": age_bracket,
        "region_income_tier": region_income_tier,
        "driver_age": driver_age,
        "years_licensed": years_licensed,
        "prior_claims_3yr": prior_claims_3yr,
        "credit_stability_score": credit_stability_score,
        "vehicle_value_usd": vehicle_value_usd,
        "vehicle_safety_rating": vehicle_safety_rating,
        "annual_mileage_km": annual_mileage_km,
        "policy_tenure_years": policy_tenure_years,
        "rt_telematics_score": rt_telematics_score,
        "rt_harsh_braking_per_100km": rt_harsh_braking_per_100km,
        "rt_night_driving_ratio": rt_night_driving_ratio,
        "rt_overspeed_ratio": rt_overspeed_ratio,
        "rt_weather_route_hazard_idx": rt_weather_route_hazard_idx,
        "true_claim_prob": true_claim_prob.round(4),
        "expected_severity_if_claim": expected_severity_if_claim,
        "claim_occurred": claim_occurred,
        "observed_claim_loss": observed_claim_loss,
    })

    return df


def ensure_datasets_saved(data_dir: str = "data") -> dict:
    """Generates and saves reference and monitoring datasets to disk for reproducibility."""
    os.makedirs(data_dir, exist_ok=True)
    ref_path = os.path.join(data_dir, "historical_realtime_underwriting.csv")
    drift_path = os.path.join(data_dir, "production_stream_q3_drift.csv")

    df_ref = generate_underwriting_dataset(n_samples=3500, random_state=42, drift_scenario="baseline")
    df_drift = generate_underwriting_dataset(n_samples=1200, random_state=99, drift_scenario="q3_severe_drift")

    df_ref.to_csv(ref_path, index=False)
    df_drift.to_csv(drift_path, index=False)

    return {
        "reference_path": ref_path,
        "drift_path": drift_path,
        "df_ref": df_ref,
        "df_drift": df_drift,
    }
