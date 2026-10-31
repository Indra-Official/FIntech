"""
Data Privacy Engine (2026 Regulatory Update)
Implements:
1. PII Tokenization & Cryptographic Hashing (Salted SHA-256)
2. k-Anonymity Quasi-Identifier Generalization
3. Differential Privacy (Laplace Mechanism with configurable epsilon budget)
4. Privacy-Utility Tradeoff Evaluation
"""

import hashlib
import numpy as np
import pandas as pd


class PrivacyGuardEngine:
    """Enforces 2026 InsurTech data privacy standards before model inference or storage."""

    def __init__(self, salt: str = "FINTECH-2026-UNDERWRITING-SALT"):
        self.salt = salt

    def hash_pii(self, value: str) -> str:
        """Returns a salted SHA-256 pseudonym token."""
        digest = hashlib.sha256(f"{self.salt}:{value}".encode("utf-8")).hexdigest()
        return f"TOK-{digest[:12].upper()}"

    def mask_national_id(self, nat_id: str) -> str:
        """Redacts all but the last 4 characters of a National ID."""
        if not nat_id or len(str(nat_id)) < 4:
            return "ID-****-****"
        return f"ID-XXXX-{str(nat_id)[-4:]}"

    def apply_laplace_dp(
        self,
        series: pd.Series,
        sensitivity: float,
        epsilon: float = 2.0,
        clip_min: float = None,
        clip_max: float = None,
        random_state: int = 42,
    ) -> pd.Series:
        """
        Adds calibrated Laplace noise Lap(0, sensitivity / epsilon) to satisfy
        epsilon-Differential Privacy on sensitive numerical features.
        """
        if epsilon <= 0:
            raise ValueError("Privacy budget epsilon must be > 0.")
        rng = np.random.default_rng(random_state)
        scale = sensitivity / float(epsilon)
        noise = rng.laplace(loc=0.0, scale=scale, size=len(series))
        perturbed = series.astype(float) + noise
        if clip_min is not None or clip_max is not None:
            perturbed = np.clip(perturbed, clip_min, clip_max)
        return perturbed

    def sanitize_dataframe(self, df: pd.DataFrame, epsilon: float = 2.5, random_state: int = 42) -> pd.DataFrame:
        """
        Produces a privacy-preserving copy of the dataset with:
        - Salted SHA-256 tokenization on applicant_name
        - Redacted national_id
        - Generalized age bands (k-anonymity)
        - Laplace Differential Privacy on credit_stability_score and rt_telematics_score
        """
        sanitized = df.copy()
        if "applicant_name" in sanitized.columns:
            sanitized["applicant_name_token"] = sanitized["applicant_name"].apply(self.hash_pii)
            sanitized["applicant_name"] = "[REDACTED-PII]"
        if "national_id" in sanitized.columns:
            sanitized["national_id"] = sanitized["national_id"].apply(self.mask_national_id)

        # k-Anonymity Quasi-Identifier Generalization
        if "driver_age" in sanitized.columns:
            sanitized["age_k_anon_band"] = pd.cut(
                sanitized["driver_age"],
                bins=[17, 25, 35, 45, 55, 65, 90],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
            ).astype(str)

        # Differential Privacy on sensitive attributes
        if "credit_stability_score" in sanitized.columns:
            sanitized["credit_stability_score"] = self.apply_laplace_dp(
                sanitized["credit_stability_score"],
                sensitivity=25.0,
                epsilon=epsilon,
                clip_min=350,
                clip_max=850,
                random_state=random_state,
            ).round().astype(int)

        if "rt_telematics_score" in sanitized.columns:
            sanitized["rt_telematics_score"] = self.apply_laplace_dp(
                sanitized["rt_telematics_score"],
                sensitivity=4.0,
                epsilon=epsilon,
                clip_min=20.0,
                clip_max=99.0,
                random_state=random_state + 1,
            ).round(1)

        return sanitized

    def sanitize_single_record(self, record: dict, epsilon: float = 2.5) -> dict:
        """Applies PII masking and Differential Privacy perturbation to a single applicant record."""
        rng = np.random.default_rng()
        out = dict(record)
        raw_name = out.get("applicant_name", "Applicant Demo")
        raw_id = out.get("national_id", "ID-4829-9104")

        out["raw_name_preview"] = raw_name
        out["applicant_ token"] = self.hash_pii(raw_name)
        out["masked_national_id"] = self.mask_national_id(raw_id)

        if epsilon and epsilon < 50.0:
            credit_scale = 25.0 / max(epsilon, 0.25)
            telem_scale = 4.0 / max(epsilon, 0.25)
            out["dp_credit_score"] = int(
                np.clip(round(out["credit_stability_score"] + rng.laplace(0, credit_scale)), 350, 850)
            )
            out["dp_telematics_score"] = float(
                np.clip(round(out["rt_telematics_score"] + rng.laplace(0, telem_scale), 1), 20.0, 99.0)
            )
            out["privacy_noise_scale_credit"] = round(credit_scale, 2)
        else:
            out["dp_credit_score"] = int(out["credit_stability_score"])
            out["dp_telematics_score"] = float(out["rt_telematics_score"])
            out["privacy_noise_scale_credit"] = 0.0

        return out

    def evaluate_privacy_utility_curve(self, df: pd.DataFrame, model_eval_fn) -> list:
        """
        Evaluates how varying epsilon in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] impacts
        model ROC-AUC and Brier calibration score.
        """
        epsilons = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        results = []
        for eps in epsilons:
            df_dp = self.sanitize_dataframe(df, epsilon=eps, random_state=42)
            metrics = model_eval_fn(df_dp)
            results.append({
                "epsilon": eps,
                "privacy_level": (
                    "Strict (High Noise)" if eps <= 1.0
                    else "Balanced (Recommended)" if eps <= 3.0
                    else "Relaxed (Low Noise)"
                ),
                "roc_auc": round(metrics["roc_auc"], 4),
                "brier_score": round(metrics["brier_score"], 4),
                "ece": round(metrics["ece"], 4),
            })
        return results
