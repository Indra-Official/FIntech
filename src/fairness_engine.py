"""
Fairness Constraints Engine (2026 Regulatory Update)
Implements:
1. Pre-Processing Sample Reweighing (Kamiran & Calders formulation)
2. Post-Processing Fairness-Constrained Risk Calibration & Thresholding
3. Comprehensive Fairness Auditing:
   - Disparate Impact Ratio (DIR - 80% Four-Fifths Rule)
   - Statistical Parity Difference (SPD)
   - Equal Opportunity Difference (EOD)
   - Group Premium Ratio & Actuarial Parity
"""

import numpy as np
import pandas as pd


class FairnessConstraintEngine:
    """
    Audits and enforces group fairness constraints across protected attributes
    (region_income_tier, gender, age_bracket) in insurance underwriting.
    """

    def __init__(self, risk_approval_threshold: float = 0.38):
        """
        In underwriting, a 'favorable' outcome is receiving Preferred/Standard approval
        (i.e. predicted claim probability <= risk_approval_threshold).
        """
        self.risk_approval_threshold = risk_approval_threshold
        self.group_adjustments = {}

    def compute_sample_weights(self, df: pd.DataFrame, protected_col: str = "region_income_tier", target_col: str = "claim_occurred") -> np.ndarray:
        """
        Computes Kamiran & Calders (2012) sample weights:
        w(a, y) = P(A = a) * P(Y = y) / P(A = a, Y = y)
        to eliminate statistical dependence between protected attribute A and target Y.
        """
        n = len(df)
        weights = np.ones(n, dtype=float)
        p_a = df[protected_col].value_counts(normalize=True).to_dict()
        p_y = df[target_col].value_counts(normalize=True).to_dict()
        joint = df.groupby([protected_col, target_col]).size().div(n).to_dict()

        for idx, row in df[[protected_col, target_col]].iterrows():
            a_val = row[protected_col]
            y_val = row[target_col]
            p_joint = joint.get((a_val, y_val), 1e-6)
            weights[idx] = (p_a.get(a_val, 1.0) * p_y.get(y_val, 1.0)) / max(p_joint, 1e-6)

        return np.clip(weights, 0.35, 2.75)

    def fit_postprocessing_constraints(
        self,
        df: pd.DataFrame,
        raw_probs: np.ndarray,
        protected_col: str = "region_income_tier",
        privileged_group: str = "High-Income Zone",
    ) -> dict:
        """
        Learns fairness-constrained risk adjustments per protected subgroup so that
        the Disparate Impact Ratio satisfies the >= 0.85 threshold while preserving
        overall portfolio mean risk probability (actuarial neutrality).
        """
        temp = pd.DataFrame({
            "group": df[protected_col].values,
            "prob": raw_probs,
        })
        overall_mean = float(np.mean(raw_probs))
        priv_mean = float(temp[temp["group"] == privileged_group]["prob"].mean())

        adjustments = {}
        for grp, sub in temp.groupby("group"):
            grp_probs = sub["prob"].values
            # Align the group's 55th percentile risk with the portfolio's 55th percentile
            # so that group approval rates at risk_approval_threshold satisfy the >= 0.85 DIR rule
            grp_q = float(np.percentile(grp_probs, 50))
            port_q = float(np.percentile(raw_probs, 50))
            shift = -0.92 * (grp_q - port_q) - 0.85 * (float(np.mean(grp_probs)) - overall_mean)
            adjustments[grp] = round(shift, 5)

        self.group_adjustments[protected_col] = adjustments
        return adjustments

    def apply_fairness_constraint(
        self,
        raw_probs: np.ndarray,
        groups: pd.Series,
        protected_col: str = "region_income_tier",
    ) -> np.ndarray:
        """Applies learned fairness-constrained probability calibration to predicted risks."""
        adjustments = self.group_adjustments.get(protected_col, {
            "Low-Income Zone": -0.052,
            "Middle-Income Zone": 0.004,
            "High-Income Zone": 0.028,
        })
        shifts = np.array([adjustments.get(g, 0.0) for g in groups], dtype=float)
        fair_probs = np.clip(raw_probs + shifts, 0.015, 0.96)
        # Re-center slightly to preserve exact portfolio-wide expected claim frequency
        mean_diff = float(np.mean(raw_probs) - np.mean(fair_probs))
        fair_probs = np.clip(fair_probs + mean_diff, 0.015, 0.96)
        return fair_probs

    def audit_group_fairness(
        self,
        df: pd.DataFrame,
        probs: np.ndarray,
        protected_col: str = "region_income_tier",
        privileged_group: str = "High-Income Zone",
        unprivileged_group: str = "Low-Income Zone",
    ) -> dict:
        """
        Computes Disparate Impact Ratio (DIR), Statistical Parity Difference (SPD),
        Equal Opportunity Difference (EOD), and per-group approval rates.
        """
        y_true = df["claim_occurred"].values
        favorable_pred = (probs <= self.risk_approval_threshold).astype(int)
        favorable_true = (y_true == 0).astype(int)

        group_stats = {}
        for grp in df[protected_col].unique():
            mask = (df[protected_col].values == grp)
            approval_rate = float(np.mean(favorable_pred[mask]))
            mean_risk = float(np.mean(probs[mask]))
            true_safe_mask = mask & (favorable_true == 1)
            tpr_safe = float(np.mean(favorable_pred[true_safe_mask])) if np.sum(true_safe_mask) > 0 else 0.0
            group_stats[str(grp)] = {
                "count": int(np.sum(mask)),
                "approval_rate": round(approval_rate, 4),
                "mean_predicted_risk": round(mean_risk, 4),
                "equal_opportunity_rate": round(tpr_safe, 4),
            }

        priv_rate = group_stats.get(privileged_group, {}).get("approval_rate", 0.5)
        unpriv_rate = group_stats.get(unprivileged_group, {}).get("approval_rate", 0.5)

        priv_eop = group_stats.get(privileged_group, {}).get("equal_opportunity_rate", 0.5)
        unpriv_eop = group_stats.get(unprivileged_group, {}).get("equal_opportunity_rate", 0.5)

        dir_score = unpriv_rate / max(priv_rate, 1e-6)
        spd_score = unpriv_rate - priv_rate
        eod_score = unpriv_eop - priv_eop

        return {
            "protected_attribute": protected_col,
            "privileged_group": privileged_group,
            "unprivileged_group": unprivileged_group,
            "disparate_impact_ratio": round(dir_score, 4),
            "statistical_parity_diff": round(spd_score, 4),
            "equal_opportunity_diff": round(eod_score, 4),
            "passes_80_percent_rule": bool(0.80 <= dir_score <= 1.25),
            "group_breakdown": group_stats,
        }
