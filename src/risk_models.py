"""
Calibrated Risk Models, Actuarial Pricing & Loss Forecasting Engine (Statement #31 - 2026 Update)
Implements:
1. Uncalibrated Base Ensemble (Random Forest / Gradient Boosted Trees)
2. Probability Calibration: Platt Scaling (Sigmoid) & Isotonic Regression
3. Expected Calibration Error (ECE), Brier Score, Log-Loss & Reliability Diagrams
4. Claim Severity Regression & Portfolio Loss Forecasting (VaR / Expected Loss)
5. Real-Time Dynamic Underwriting Decision & Explainable Premium Breakdown
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss, mean_absolute_error, r2_score

from src.data_pipeline import FEATURE_COLUMNS
from src.fairness_engine import FairnessConstraintEngine
from src.privacy_engine import PrivacyGuardEngine


class PredictiveUnderwritingPipeline:
    """End-to-end Calibrated Risk, Pricing, Fairness, and Loss Forecasting Pipeline."""

    def __init__(self):
        self.feature_cols = FEATURE_COLUMNS
        self.base_clf = None
        self.platt_clf = None
        self.isotonic_clf = None
        self.severity_reg = None
        self.fairness_engine = FairnessConstraintEngine(risk_approval_threshold=0.38)
        self.privacy_engine = PrivacyGuardEngine()
        self.training_summary = {}
        self.feature_medians = {}

    @staticmethod
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE):
        ECE = sum_{b=1..B} (|B_b| / N) * | acc(B_b) - conf(B_b) |
        """
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
        ece = 0.0
        n = len(y_true)
        for b in range(n_bins):
            mask = bin_ids == b
            if np.any(mask):
                emp_freq = float(np.mean(y_true[mask]))
                mean_pred = float(np.mean(y_prob[mask]))
                ece += (np.sum(mask) / n) * abs(emp_freq - mean_pred)
        return round(float(ece), 5)

    def evaluate_predictions(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        """Returns ROC-AUC, Brier Score, Log-Loss, and Expected Calibration Error (ECE)."""
        probs_clipped = np.clip(y_prob, 1e-5, 1.0 - 1e-5)
        return {
            "roc_auc": round(float(roc_auc_score(y_true, probs_clipped)), 4),
            "brier_score": round(float(brier_score_loss(y_true, probs_clipped)), 4),
            "log_loss": round(float(log_loss(y_true, probs_clipped)), 4),
            "ece": round(self.compute_ece(y_true, probs_clipped, n_bins=10), 4),
        }

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Trains:
        1. Uncalibrated RandomForestClassifier (intentionally pushed to high variance/depth to highlight calibration gain)
        2. Platt Scaled (Sigmoid) Calibrated Classifier
        3. Isotonic Regression Calibrated Classifier (with Fairness sample weights)
        4. Claim Severity GradientBoostingRegressor
        """
        X = df[self.feature_cols].copy()
        y = df["claim_occurred"].values
        y_sev = df["expected_severity_if_claim"].values

        for col in self.feature_cols:
            self.feature_medians[col] = float(X[col].median())

        # Compute Kamiran & Calders Fairness Sample Weights
        sample_weights = self.fairness_engine.compute_sample_weights(
            df, protected_col="region_income_tier", target_col="claim_occurred"
        )

        X_train, X_test, y_train, y_test, w_train, w_test, idx_train, idx_test = train_test_split(
            X, y, sample_weights, np.arange(len(df)),
            test_size=0.28, random_state=42, stratify=y
        )

        # 1. Base Uncalibrated Classifier
        self.base_clf = RandomForestClassifier(
            n_estimators=140,
            max_depth=12,
            min_samples_leaf=4,
            class_weight={0: 1.0, 1: 2.3},  # Cost-sensitive weighting causes uncalibrated probability distortion
            random_state=42,
        )
        self.base_clf.fit(X_train, y_train)

        # 2. Platt Scaling (Sigmoid Calibration)
        self.platt_clf = CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=140, max_depth=12, min_samples_leaf=4,
                class_weight={0: 1.0, 1: 2.3}, random_state=42
            ),
            method="sigmoid",
            cv=5,
        )
        self.platt_clf.fit(X_train, y_train, sample_weight=w_train)

        # 3. Isotonic Calibration
        self.isotonic_clf = CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=140, max_depth=12, min_samples_leaf=4,
                class_weight={0: 1.0, 1: 2.3}, random_state=42
            ),
            method="isotonic",
            cv=5,
        )
        self.isotonic_clf.fit(X_train, y_train, sample_weight=w_train)

        # 4. Claim Severity Regressor (Loss Forecasting Model)
        self.severity_reg = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.06, max_depth=4, random_state=42
        )
        self.severity_reg.fit(X_train, y_sev[idx_train])

        # Evaluate Calibration on Holdout Test Set
        prob_uncal = self.base_clf.predict_proba(X_test)[:, 1]
        prob_platt = self.platt_clf.predict_proba(X_test)[:, 1]
        prob_iso = self.isotonic_clf.predict_proba(X_test)[:, 1]

        metrics_uncal = self.evaluate_predictions(y_test, prob_uncal)
        metrics_platt = self.evaluate_predictions(y_test, prob_platt)
        metrics_iso = self.evaluate_predictions(y_test, prob_iso)

        # Reliability Curves (8 bins for clean visualization)
        frac_uncal, mean_uncal = calibration_curve(y_test, prob_uncal, n_bins=8, strategy="uniform")
        frac_platt, mean_platt = calibration_curve(y_test, prob_platt, n_bins=8, strategy="uniform")
        frac_iso, mean_iso = calibration_curve(y_test, prob_iso, n_bins=8, strategy="uniform")

        # Fit Post-Processing Fairness Constraint on Test Set
        df_test = df.iloc[idx_test].reset_index(drop=True)
        self.fairness_engine.fit_postprocessing_constraints(
            df_test, prob_iso, protected_col="region_income_tier"
        )
        prob_fair = self.fairness_engine.apply_fairness_constraint(
            prob_iso, df_test["region_income_tier"], protected_col="region_income_tier"
        )

        fairness_before = self.fairness_engine.audit_group_fairness(
            df_test, prob_uncal, protected_col="region_income_tier"
        )
        fairness_after = self.fairness_engine.audit_group_fairness(
            df_test, prob_fair, protected_col="region_income_tier"
        )

        # Gender fairness check as well
        gender_fairness = self.fairness_engine.audit_group_fairness(
            df_test, prob_fair, protected_col="gender",
            privileged_group="Male", unprivileged_group="Female"
        )

        # Severity & Portfolio Loss Forecasting Metrics
        sev_pred_test = np.clip(self.severity_reg.predict(X_test), 500.0, 90000.0)
        sev_mae = float(mean_absolute_error(y_sev[idx_test], sev_pred_test))
        sev_r2 = float(r2_score(y_sev[idx_test], sev_pred_test))

        # Expected Loss (Pure Premium) across test portfolio
        pure_premiums_test = prob_fair * sev_pred_test
        actual_losses_test = df_test["observed_claim_loss"].values

        # Feature Importances from Base RF + Severity Regressor
        rf_imp = self.base_clf.feature_importances_
        feature_importance_list = sorted(
            [
                {"feature": col, "importance": round(float(imp * 100.0), 2)}
                for col, imp in zip(self.feature_cols, rf_imp)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )

        self.training_summary = {
            "test_size": int(len(y_test)),
            "calibration_comparison": {
                "uncalibrated": metrics_uncal,
                "platt_sigmoid": metrics_platt,
                "isotonic_calibrated": metrics_iso,
            },
            "reliability_curves": {
                "uncalibrated": {
                    "mean_predicted": [round(float(x), 4) for x in mean_uncal],
                    "fraction_positives": [round(float(x), 4) for x in frac_uncal],
                },
                "platt_sigmoid": {
                    "mean_predicted": [round(float(x), 4) for x in mean_platt],
                    "fraction_positives": [round(float(x), 4) for x in frac_platt],
                },
                "isotonic_calibrated": {
                    "mean_predicted": [round(float(x), 4) for x in mean_iso],
                    "fraction_positives": [round(float(x), 4) for x in frac_iso],
                },
            },
            "fairness_audit": {
                "before_mitigation": fairness_before,
                "after_mitigation": fairness_after,
                "gender_audit": gender_fairness,
            },
            "loss_forecasting": {
                "severity_mae_usd": round(sev_mae, 2),
                "severity_r2": round(sev_r2, 4),
                "portfolio_expected_loss_usd": round(float(np.sum(pure_premiums_test)), 2),
                "portfolio_actual_loss_usd": round(float(np.sum(actual_losses_test)), 2),
                "avg_pure_premium_usd": round(float(np.mean(pure_premiums_test)), 2),
                "loss_forecast_accuracy_pct": round(
                    100.0 * (1.0 - abs(np.sum(pure_premiums_test) - np.sum(actual_losses_test)) / max(np.sum(actual_losses_test), 1.0)),
                    2,
                ),
            },
            "feature_importances": feature_importance_list,
        }
        return self.training_summary

    def predict_calibrated_prob(self, df: pd.DataFrame, apply_fairness: bool = True) -> np.ndarray:
        """Predicts calibrated claim probabilities on a DataFrame."""
        X = df[self.feature_cols]
        probs = self.isotonic_clf.predict_proba(X)[:, 1]
        if apply_fairness and "region_income_tier" in df.columns:
            probs = self.fairness_engine.apply_fairness_constraint(
                probs, df["region_income_tier"], protected_col="region_income_tier"
            )
        return probs

    def underwrite_single_applicant(
        self,
        applicant_data: dict,
        enable_fairness: bool = True,
        enable_privacy: bool = True,
        privacy_epsilon: float = 2.5,
    ) -> dict:
        """
        Real-Time Predictive Underwriting & Actuarial Pricing Engine for a single policy quote.
        Returns:
        - Uncalibrated vs Calibrated vs Fairness-Constrained Claim Probability
        - Expected Claim Severity & Pure Premium (Expected Loss)
        - Real-Time Telematics Dynamic Discount / Loading
        - Final Annual & Monthly Technical Premium
        - Underwriting Decision Tier & Explainable Risk Drivers
        - Privacy Sanitization Metadata
        """
        privacy_meta = self.privacy_engine.sanitize_single_record(
            applicant_data, epsilon=privacy_epsilon if enable_privacy else 100.0
        )

        # Build single-row DataFrame for inference
        row = {}
        for col in self.feature_cols:
            row[col] = float(applicant_data.get(col, self.feature_medians.get(col, 0.0)))

        if enable_privacy:
            row["credit_stability_score"] = float(privacy_meta["dp_credit_score"])
            row["rt_telematics_score"] = float(privacy_meta["dp_telematics_score"])

        X_single = pd.DataFrame([row])[self.feature_cols]

        uncal_prob = float(self.base_clf.predict_proba(X_single)[:, 1][0])
        platt_prob = float(self.platt_clf.predict_proba(X_single)[:, 1][0])
        iso_prob = float(self.isotonic_clf.predict_proba(X_single)[:, 1][0])

        region_tier = applicant_data.get("region_income_tier", "Middle-Income Zone")
        if enable_fairness:
            final_prob = float(
                self.fairness_engine.apply_fairness_constraint(
                    np.array([iso_prob]), pd.Series([region_tier]), protected_col="region_income_tier"
                )[0]
            )
            # Ensure exact single-record adjustment is visible
            grp_shift = self.fairness_engine.group_adjustments.get("region_income_tier", {}).get(region_tier, 0.0)
            final_prob = float(np.clip(iso_prob + grp_shift, 0.02, 0.95))
        else:
            final_prob = iso_prob
            grp_shift = 0.0

        expected_severity = float(np.clip(self.severity_reg.predict(X_single)[0], 600.0, 85000.0))

        # Pure Premium = P(Claim) * E[Severity | Claim]
        pure_premium = final_prob * expected_severity

        # Real-time Telematics Dynamic Pricing Modifier (-22% discount to +35% surcharge)
        telem_score = row["rt_telematics_score"]
        telematics_modifier_pct = round((72.0 - telem_score) * 0.65, 2)
        telematics_adjustment_usd = pure_premium * (telematics_modifier_pct / 100.0)

        # Actuarial Technical Premium Formula:
        # Technical Premium = (Pure Premium + Telematics Adj + Fixed Admin Expense) / (1 - Expense & Risk Margin Ratio)
        fixed_admin_expense = 115.0
        expense_and_capital_loading_ratio = 0.24  # 16% operational + 8% solvency II risk margin
        annual_technical_premium = max(
            240.0,
            (pure_premium + telematics_adjustment_usd + fixed_admin_expense) / (1.0 - expense_and_capital_loading_ratio),
        )
        monthly_premium = annual_technical_premium / 12.0

        # Underwriting Decision Tiering
        if final_prob < 0.12 and telem_score >= 75:
            tier = "TIER 1: PREFERRED PLUS"
            decision = "INSTANT AUTO-APPROVE (Preferred Discount Applied)"
            badge_color = "emerald"
        elif final_prob < 0.24:
            tier = "TIER 2: STANDARD RISK"
            decision = "AUTO-APPROVE (Standard Actuarial Terms)"
            badge_color = "blue"
        elif final_prob < 0.40:
            tier = "TIER 3: SUBSTANDARD / RATED"
            decision = "CONDITIONAL APPROVE (Telematics Monitoring + Risk Loading)"
            badge_color = "amber"
        else:
            tier = "TIER 4: HIGH RISK / DECLINE"
            decision = "SENIOR UNDERWRITER REFERRAL / DECLINE"
            badge_color = "rose"

        # Explainable Local Factor Contributions (transparent driver breakdown)
        drivers = [
            {
                "factor": "Real-Time Telematics Score",
                "value": f"{row['rt_telematics_score']:.1f} / 100",
                "impact_pct": round((72.0 - row["rt_telematics_score"]) * 0.75, 1),
                "category": "Real-Time Stream",
            },
            {
                "factor": "Harsh Braking Frequency",
                "value": f"{row['rt_harsh_braking_per_100km']:.1f} / 100km",
                "impact_pct": round((row["rt_harsh_braking_per_100km"] - 2.8) * 3.8, 1),
                "category": "Real-Time Stream",
            },
            {
                "factor": "Overspeeding & Night Ratio",
                "value": f"{(row['rt_overspeed_ratio']*100):.1f}% / {(row['rt_night_driving_ratio']*100):.1f}%",
                "impact_pct": round(((row["rt_overspeed_ratio"] - 0.18) * 45.0 + (row["rt_night_driving_ratio"] - 0.21) * 30.0), 1),
                "category": "Real-Time Stream",
            },
            {
                "factor": "Prior Claims (3-Year History)",
                "value": f"{int(row['prior_claims_3yr'])} claims",
                "impact_pct": round((row["prior_claims_3yr"] - 0.38) * 14.5, 1),
                "category": "Historical Data",
            },
            {
                "factor": "Credit Stability Score (DP-Protected)",
                "value": f"{int(row['credit_stability_score'])}",
                "impact_pct": round((690.0 - row["credit_stability_score"]) * 0.08, 1),
                "category": "Historical Data",
            },
            {
                "factor": "2026 Fairness Constraint Adjustment",
                "value": region_tier if enable_fairness else "Disabled",
                "impact_pct": round(grp_shift * 100.0, 1),
                "category": "2026 Governance",
            },
        ]

        return {
            "probabilities": {
                "uncalibrated_prob": round(uncal_prob, 4),
                "platt_calibrated_prob": round(platt_prob, 4),
                "isotonic_calibrated_prob": round(iso_prob, 4),
                "final_underwriting_prob": round(final_prob, 4),
                "fairness_shift_applied": round(grp_shift, 4),
            },
            "pricing": {
                "expected_claim_severity_usd": round(expected_severity, 2),
                "pure_premium_expected_loss_usd": round(pure_premium, 2),
                "telematics_modifier_pct": telematics_modifier_pct,
                "telematics_adjustment_usd": round(telematics_adjustment_usd, 2),
                "fixed_admin_expense_usd": fixed_admin_expense,
                "solvency_and_expense_loading_usd": round(
                    annual_technical_premium - (pure_premium + telematics_adjustment_usd + fixed_admin_expense), 2
                ),
                "annual_technical_premium_usd": round(annual_technical_premium, 2),
                "monthly_premium_usd": round(monthly_premium, 2),
            },
            "decision": {
                "tier": tier,
                "action": decision,
                "badge_color": badge_color,
            },
            "privacy_audit": {
                "enabled": enable_privacy,
                "epsilon": privacy_epsilon,
                "applicant_token": privacy_meta["applicant_ token"],
                "masked_national_id": privacy_meta["masked_national_id"],
                "raw_credit_score": int(applicant_data.get("credit_stability_score", 690)),
                "dp_credit_score": privacy_meta["dp_credit_score"],
                "raw_telematics_score": float(applicant_data.get("rt_telematics_score", 75.0)),
                "dp_telematics_score": privacy_meta["dp_telematics_score"],
            },
            "explainable_drivers": drivers,
        }
