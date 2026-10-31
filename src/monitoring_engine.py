"""
Model Monitoring & Drift Detection Engine (2026 Regulatory Update)
Implements:
1. Population Stability Index (PSI) per feature and on predicted risk scores
2. Two-Sample Kolmogorov-Smirnov (KS) Drift Test (statistic + p-value)
3. Calibration & Actuarial Loss Ratio Drift Monitoring across Production Windows
4. Automated Retraining & Governance Alerts
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class ModelMonitoringEngine:
    """Monitors covariate drift, prediction drift, and calibration decay in production."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Computes the Population Stability Index (PSI):
        PSI = sum( (p_curr - p_ref) * ln(p_curr / p_ref) )
        Thresholds:
          < 0.10 : Stable (Green)
          0.10 - 0.25 : Moderate Drift / Investigation Required (Amber)
          >= 0.25 : Critical Drift / Retrain Triggered (Red)
        """
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)

        quantiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.unique(np.percentile(ref, quantiles))
        if len(bins) < 3:
            bins = np.linspace(np.min(ref) - 1e-5, np.max(ref) + 1e-5, self.n_bins + 1)
        else:
            bins[0] = min(bins[0], np.min(cur)) - 1e-5
            bins[-1] = max(bins[-1], np.max(cur)) + 1e-5

        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)

        ref_prop = np.clip(ref_counts / max(len(ref), 1), 1e-4, 1.0)
        cur_prop = np.clip(cur_counts / max(len(cur), 1), 1e-4, 1.0)

        psi_val = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
        return round(max(psi_val, 0.0), 4)

    def evaluate_feature_drift(self, df_ref: pd.DataFrame, df_cur: pd.DataFrame, feature_cols: list) -> list:
        """Computes PSI and KS-test drift statistics for every model feature."""
        report = []
        for col in feature_cols:
            ref_vals = df_ref[col].values
            cur_vals = df_cur[col].values
            psi = self.compute_psi(ref_vals, cur_vals)
            ks_stat, ks_pval = ks_2samp(ref_vals, cur_vals)

            if psi >= 0.25:
                status = "CRITICAL DRIFT"
                severity = "high"
            elif psi >= 0.10:
                status = "MODERATE DRIFT"
                severity = "medium"
            else:
                status = "STABLE"
                severity = "low"

            report.append({
                "feature": col,
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_pval), 5),
                "ref_mean": round(float(np.mean(ref_vals)), 3),
                "cur_mean": round(float(np.mean(cur_vals)), 3),
                "pct_shift": round(
                    float((np.mean(cur_vals) - np.mean(ref_vals)) / (abs(np.mean(ref_vals)) + 1e-6) * 100.0),
                    2,
                ),
                "status": status,
                "severity": severity,
            })

        report.sort(key=lambda x: x["psi"], reverse=True)
        return report

    def evaluate_quarterly_timeline(self, model_pipeline, data_generator_fn, feature_cols: list) -> list:
        """
        Simulates 4 sequential operational windows (Baseline, Q1, Q2, Q3)
        to track PSI, Calibration Error (ECE), Brier Score, and Actual-to-Expected (A/E) Loss Ratio.
        """
        df_base = data_generator_fn(n_samples=1500, random_state=42, drift_scenario="baseline")
        windows = [
            ("2026-Q1 (Baseline)", "baseline", 101),
            ("2026-Q2 (Normal Stream)", "q1_normal", 102),
            ("2026-Q3 (Mild Telematics Shift)", "q2_mild_drift", 103),
            ("2026-Q4 (Severe Macro + Weather Shift)", "q3_severe_drift", 104),
        ]

        timeline = []
        base_preds = model_pipeline.predict_calibrated_prob(df_base)

        for label, scenario, seed in windows:
            df_win = data_generator_fn(n_samples=1000, random_state=seed, drift_scenario=scenario)
            preds = model_pipeline.predict_calibrated_prob(df_win)
            eval_metrics = model_pipeline.evaluate_predictions(df_win["claim_occurred"].values, preds)

            pred_psi = self.compute_psi(base_preds, preds)
            feat_psis = [self.compute_psi(df_base[c].values, df_win[c].values) for c in feature_cols]
            max_feat_psi = max(feat_psis)

            # Actual vs Expected Claim Frequency Ratio
            ae_ratio = float(np.mean(df_win["claim_occurred"].values) / max(np.mean(preds), 1e-5))

            alert = (
                "ACTION REQUIRED: Auto-Recalibration & Retrain Triggered"
                if max_feat_psi >= 0.25 or eval_metrics["ece"] > 0.045
                else "WATCH: Elevated Covariate Shift Detected"
                if max_feat_psi >= 0.10
                else "HEALTHY: Within Actuarial Guardrails"
            )

            timeline.append({
                "period": label,
                "scenario": scenario,
                "prediction_psi": round(pred_psi, 4),
                "max_feature_psi": round(max_feat_psi, 4),
                "roc_auc": round(eval_metrics["roc_auc"], 4),
                "brier_score": round(eval_metrics["brier_score"], 4),
                "ece": round(eval_metrics["ece"], 4),
                "actual_to_expected_ratio": round(ae_ratio, 3),
                "alert_status": alert,
            })

        return timeline
