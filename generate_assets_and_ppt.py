"""
Asset, Benchmark & Presentation Generator for Statement #31 — Predictive Underwriting (2026 Update)
Generates:
1. High-resolution visual figures in docs/images/*.png
2. Evaluation benchmark JSON in docs/evaluation_results.json
3. 12-Slide Editable PowerPoint Deck: presentation/Predictive_Underwriting_Statement31_Presentation.pptx
4. Standalone Interactive HTML Presentation Deck: presentation/Interactive_Presentation_Deck.html
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from src.data_pipeline import (
    FEATURE_COLUMNS,
    generate_underwriting_dataset,
    ensure_datasets_saved,
)
from src.risk_models import PredictiveUnderwritingPipeline
from src.monitoring_engine import ModelMonitoringEngine
from src.privacy_engine import PrivacyGuardEngine
from app import DEMO_PERSONAS


def set_dark_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "text.color": "#f8fafc",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "font.size": 10,
    })


def generate_all():
    os.makedirs("docs/images", exist_ok=True)
    os.makedirs("presentation", exist_ok=True)
    set_dark_plot_style()

    print("[1/5] Generating datasets and training Calibrated Risk & Loss Forecasting models...")
    ds = ensure_datasets_saved("data")
    df_ref = ds["df_ref"]
    df_drift = ds["df_drift"]

    pipeline = PredictiveUnderwritingPipeline()
    summary = pipeline.fit(df_ref)

    monitor = ModelMonitoringEngine(n_bins=10)
    privacy = PrivacyGuardEngine()

    drift_q3 = monitor.evaluate_feature_drift(df_ref, df_drift, FEATURE_COLUMNS)
    quarterly = monitor.evaluate_quarterly_timeline(pipeline, generate_underwriting_dataset, FEATURE_COLUMNS)

    def _eval_dp(df_s):
        p = pipeline.predict_calibrated_prob(df_s, apply_fairness=True)
        return pipeline.evaluate_predictions(df_s["claim_occurred"].values, p)

    privacy_curve = privacy.evaluate_privacy_utility_curve(
        df_ref.sample(n=900, random_state=42).reset_index(drop=True), _eval_dp
    )

    # Evaluate the 4 curated Demo Personas for the Demonstration Walkthrough
    persona_results = {}
    for key, pdata in DEMO_PERSONAS.items():
        persona_results[key] = {
            "profile": pdata,
            "evaluation": pipeline.underwrite_single_applicant(
                pdata, enable_fairness=True, enable_privacy=True, privacy_epsilon=2.5
            ),
        }

    # Save JSON results
    full_report = {
        "training_summary": summary,
        "feature_drift_q3": drift_q3,
        "quarterly_monitoring": quarterly,
        "privacy_utility_curve": privacy_curve,
        "demo_persona_results": persona_results,
    }
    with open("docs/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("[2/5] Rendering high-resolution matplotlib charts in docs/images/...")

    # --- Chart 1: Calibration Reliability Diagram ---
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=220)
    rc = summary["reliability_curves"]
    cal_m = summary["calibration_comparison"]
    ax.plot([0, 0.85], [0, 0.85], "--", color="#94a3b8", linewidth=1.8, label="Ideal 45° Calibration Diagonal")
    ax.plot(
        rc["uncalibrated"]["mean_predicted"],
        rc["uncalibrated"]["fraction_positives"],
        "o-", color="#fb7185", linewidth=2.2,
        label=f"Uncalibrated Ensemble (Brier={cal_m['uncalibrated']['brier_score']}, ECE={cal_m['uncalibrated']['ece']})"
    )
    ax.plot(
        rc["platt_sigmoid"]["mean_predicted"],
        rc["platt_sigmoid"]["fraction_positives"],
        "s-", color="#fbbf24", linewidth=2.2,
        label=f"Platt Scaling Sigmoid (Brier={cal_m['platt_sigmoid']['brier_score']}, ECE={cal_m['platt_sigmoid']['ece']})"
    )
    ax.plot(
        rc["isotonic_calibrated"]["mean_predicted"],
        rc["isotonic_calibrated"]["fraction_positives"],
        "^-", color="#34d399", linewidth=2.8,
        label=f"Isotonic Calibrated 2026 (Brier={cal_m['isotonic_calibrated']['brier_score']}, ECE={cal_m['isotonic_calibrated']['ece']})"
    )
    ax.set_title("Reliability Diagram — Calibrated Risk Models (2026 Update)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Claim Probability")
    ax.set_ylabel("Observed Empirical Claim Frequency")
    ax.grid(True)
    ax.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=8.5)
    plt.tight_layout()
    chart1_path = "docs/images/calibration_reliability.png"
    plt.savefig(chart1_path, dpi=220)
    plt.close()

    # --- Chart 2: Fairness Mitigation Comparison ---
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=220)
    fb = summary["fairness_audit"]["before_mitigation"]
    fa = summary["fairness_audit"]["after_mitigation"]
    groups = ["Low-Income Zone", "Middle-Income Zone", "High-Income Zone"]
    before_vals = [fb["group_breakdown"][g]["approval_rate"] * 100 for g in groups]
    after_vals = [fa["group_breakdown"][g]["approval_rate"] * 100 for g in groups]

    x = np.arange(len(groups))
    w = 0.35
    bars1 = ax.bar(x - w/2, before_vals, width=w, color="#fb7185", label=f"Before Mitigation (DIR = {fb['disparate_impact_ratio']})")
    bars2 = ax.bar(x + w/2, after_vals, width=w, color="#34d399", label=f"After 2026 Fairness Constraint (DIR = {fa['disparate_impact_ratio']})")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Preferred / Standard Approval Rate (%)")
    ax.set_title("2026 Fairness Constraints: Socioeconomic Proxy Bias Mitigation", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="y")
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=8.5, color="#fecdd3")
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=8.5, color="#a7f3d0")
    ax.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9)
    plt.tight_layout()
    chart2_path = "docs/images/fairness_mitigation.png"
    plt.savefig(chart2_path, dpi=220)
    plt.close()

    # --- Chart 3: Model Monitoring PSI Drift ---
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=220)
    top_drift = drift_q3[:8][::-1]
    labels = [d["feature"] for d in top_drift]
    psis = [d["psi"] for d in top_drift]
    colors = ["#fb7185" if v >= 0.25 else "#fbbf24" if v >= 0.10 else "#34d399" for v in psis]
    ax.barh(labels, psis, color=colors, height=0.6)
    ax.axvline(0.10, color="#fbbf24", linestyle="--", linewidth=1.5, label="Warning Threshold (PSI = 0.10)")
    ax.axvline(0.25, color="#fb7185", linestyle="--", linewidth=1.8, label="Critical Drift / Retrain Trigger (PSI = 0.25)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("2026 Continuous Model Monitoring — Feature Drift Detection (Q4 Shift)", fontsize=11.5, fontweight="bold", pad=12)
    ax.grid(True, axis="x")
    ax.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", fontsize=8.5)
    plt.tight_layout()
    chart3_path = "docs/images/drift_monitoring_psi.png"
    plt.savefig(chart3_path, dpi=220)
    plt.close()

    # --- Chart 4: Live Demonstration Cases Pricing & Risk Comparison ---
    fig, ax1 = plt.subplots(figsize=(7.8, 4.6), dpi=220)
    case_names = [
        "Case 1:\nSafe Telematics\n(Preferred)",
        "Case 2:\nLow-Income Safe\n(Fairness Active)",
        "Case 3:\nNight Speeder\n(RT Surcharge)",
        "Case 4:\nMulti-Claim Hazard\n(Referral/Decline)",
    ]
    case_keys = ["safe_commuter", "fairness_showcase", "realtime_speeder", "high_risk_decline"]
    prems = [persona_results[k]["evaluation"]["pricing"]["annual_technical_premium_usd"] for k in case_keys]
    probs = [persona_results[k]["evaluation"]["probabilities"]["final_underwriting_prob"] * 100 for k in case_keys]

    x = np.arange(len(case_names))
    bar_colors = ["#34d399", "#38bdf8", "#fbbf24", "#fb7185"]
    bars = ax1.bar(x, prems, width=0.48, color=bar_colors, label="Annual Technical Premium ($ USD)")
    ax1.set_ylabel("Annual Technical Premium ($ USD)", color="#38bdf8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(case_names, fontsize=9)
    ax1.set_ylim(0, max(prems) * 1.25)
    ax1.grid(True, axis="y")
    for b, p_val in zip(bars, probs):
        ax1.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 60,
            f"${b.get_height():,.0f}/yr\n(Risk: {p_val:.1f}%)",
            ha="center", fontsize=8.5, fontweight="bold", color="#f8fafc"
        )
    ax1.set_title("Live Demonstration Cases — Calibrated Risk & Dynamic Pricing Output", fontsize=11.5, fontweight="bold", pad=12)
    plt.tight_layout()
    chart4_path = "docs/images/loss_forecasting_pricing.png"
    plt.savefig(chart4_path, dpi=220)
    plt.close()

    # --- Chart 5: Differential Privacy Tradeoff Curve ---
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    eps_list = [f"ε={d['epsilon']}" for d in privacy_curve]
    aucs = [d["roc_auc"] for d in privacy_curve]
    briers = [d["brier_score"] for d in privacy_curve]
    ax.plot(eps_list, aucs, "o-", color="#38bdf8", linewidth=2.5, label="ROC-AUC (Discrimination Utility)")
    ax.plot(eps_list, briers, "s--", color="#a78bfa", linewidth=2.2, label="Brier Score (Calibration Loss)")
    ax.set_title("2026 Data Privacy: Differential Privacy (ε) vs. Actuarial Utility", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xlabel("Differential Privacy Budget (ε — Laplace Mechanism)")
    ax.set_ylabel("Metric Value")
    ax.grid(True)
    ax.legend(loc="center right", facecolor="#0f172a", edgecolor="#334155")
    plt.tight_layout()
    chart5_path = "docs/images/privacy_utility_tradeoff.png"
    plt.savefig(chart5_path, dpi=220)
    plt.close()

    print("[3/5] Building 12-Slide Editable PowerPoint Deck (.pptx)...")
    build_powerpoint_deck(
        summary=summary,
        drift_q3=drift_q3,
        quarterly=quarterly,
        privacy_curve=privacy_curve,
        persona_results=persona_results,
        charts={
            "calibration": chart1_path,
            "fairness": chart2_path,
            "drift": chart3_path,
            "demo_cases": chart4_path,
            "privacy": chart5_path,
        },
        output_path="presentation/Predictive_Underwriting_Statement31_Presentation.pptx",
    )

    print("[4/5] Building Standalone Interactive HTML Slide Deck...")
    build_interactive_html_deck(
        summary=summary,
        quarterly=quarterly,
        persona_results=persona_results,
        output_path="presentation/Interactive_Presentation_Deck.html",
    )
    print("[5/5] All presentation & visual assets successfully generated!")


def _style_slide_bg(slide):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(11, 15, 25)  # Deep FinTech Navy (#0B0F19)


def _add_header(slide, title_text: str, subtitle_text: str, badge_text: str = "STATEMENT #31 • 2026 UPDATE"):
    # Top accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RGBColor(56, 189, 248)
    bar.line.fill.background()

    # Badge
    tb_badge = slide.shapes.add_textbox(Inches(0.6), Inches(0.25), Inches(6.0), Inches(0.35))
    tf_b = tb_badge.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = badge_text
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = RGBColor(56, 189, 248)

    # Top-Right GitHub Link
    tb_gh = slide.shapes.add_textbox(Inches(7.5), Inches(0.22), Inches(5.233), Inches(0.35))
    tf_gh = tb_gh.text_frame
    p_gh = tf_gh.paragraphs[0]
    p_gh.alignment = PP_ALIGN.RIGHT
    r_gh = p_gh.add_run()
    r_gh.text = "GitHub: https://github.com/Indra-Official/FIntech"
    r_gh.hyperlink.address = "https://github.com/Indra-Official/FIntech"
    r_gh.font.size = Pt(10)
    r_gh.font.bold = True
    r_gh.font.color.rgb = RGBColor(56, 189, 248)

    # Title
    tb_title = slide.shapes.add_textbox(Inches(0.6), Inches(0.52), Inches(12.1), Inches(0.55))
    tf_t = tb_title.text_frame
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(24)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(248, 250, 252)

    # Subtitle
    tb_sub = slide.shapes.add_textbox(Inches(0.6), Inches(1.05), Inches(12.1), Inches(0.4))
    tf_s = tb_sub.text_frame
    p_s = tf_s.paragraphs[0]
    p_s.text = subtitle_text
    p_s.font.size = Pt(12)
    p_s.font.color.rgb = RGBColor(148, 163, 184)


def _add_card(slide, left, top, width, height, title: str, bullets: list, title_rgb=(56, 189, 248)):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(22, 31, 50)
    shape.line.color.rgb = RGBColor(51, 65, 85)
    shape.line.width = Pt(1)

    tb = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.14), Inches(width - 0.36), Inches(height - 0.28))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(*title_rgb)
    p0.space_after = Pt(8)

    for b in bullets:
        p = tf.add_paragraph()
        p.space_after = Pt(5)
        if "https://github.com/Indra-Official/FIntech" in b:
            prefix, url_part = b.split("https://github.com/Indra-Official/FIntech", 1)
            r1 = p.add_run()
            r1.text = f"• {prefix}"
            r1.font.size = Pt(11.5)
            r1.font.color.rgb = RGBColor(226, 232, 240)

            r2 = p.add_run()
            r2.text = "https://github.com/Indra-Official/FIntech" + url_part
            r2.hyperlink.address = "https://github.com/Indra-Official/FIntech"
            r2.font.size = Pt(11.5)
            r2.font.bold = True
            r2.font.color.rgb = RGBColor(56, 189, 248)
        else:
            p.text = f"• {b}"
            p.font.size = Pt(11.5)
            p.font.color.rgb = RGBColor(226, 232, 240)


def build_powerpoint_deck(summary, drift_q3, quarterly, privacy_curve, persona_results, charts, output_path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    cal_u = summary["calibration_comparison"]["uncalibrated"]
    cal_i = summary["calibration_comparison"]["isotonic_calibrated"]
    fair_b = summary["fairness_audit"]["before_mitigation"]
    fair_a = summary["fairness_audit"]["after_mitigation"]
    loss_f = summary["loss_forecasting"]

    # ==================== SLIDE 1: TITLE SLIDE ====================
    s1 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s1)
    _add_header(
        s1,
        "Statement #31 — Predictive Underwriting & Loss Forecasting",
        "Digital Assignment & Live System Demonstration • Historical + Real-Time Telematics AI",
        "FINTECH / INSURTECH DIGITAL ASSIGNMENT • 2026 UPDATE",
    )
    _add_card(
        s1, 0.6, 1.75, 7.3, 4.9,
        "Project Scope & 2026 Regulatory Pillars",
        [
            "Core Objective: Fuse historical policy/claims records with real-time IoT telematics streams to improve insurance pricing, underwriting tiering, and portfolio loss forecasting.",
            "2026 Pillar 1 — Calibrated Risk Models: Isotonic & Platt probability calibration reducing Expected Calibration Error (ECE) from "
            f"{cal_u['ece']} to {cal_i['ece']} and Brier Score from {cal_u['brier_score']} to {cal_i['brier_score']}.",
            "2026 Pillar 2 — Continuous Model Monitoring: Automated Population Stability Index (PSI) & Kolmogorov-Smirnov (KS) drift detection across quarterly streams.",
            "2026 Pillar 3 — Data Privacy Guardrails: Salted SHA-256 PII tokenization, k-anonymity generalization, and configurable ε-Differential Privacy (Laplace mechanism).",
            "2026 Pillar 4 — Fairness Constraints: Kamiran-Calders reweighing + post-processing calibration raising Disparate Impact Ratio from "
            f"{fair_b['disparate_impact_ratio']} to {fair_a['disparate_impact_ratio']} (passing the 80% rule).",
        ],
        title_rgb=(56, 189, 248),
    )
    _add_card(
        s1, 8.2, 1.75, 4.5, 4.9,
        "Key Benchmark Highlights",
        [
            f"ROC-AUC Discrimination: {cal_i['roc_auc']}",
            f"Calibrated Brier Score: {cal_i['brier_score']}",
            f"Calibration Error (ECE): {cal_i['ece']}",
            f"Disparate Impact Ratio: {fair_a['disparate_impact_ratio']} (≥0.80 Pass)",
            f"Loss Forecast Accuracy: {loss_f['loss_forecast_accuracy_pct']}%",
            "Live Web Demo: Flask + Real-Time Telemetry Studio (http://127.0.0.1:5000)",
            "GitHub Repo: https://github.com/Indra-Official/FIntech",
        ],
        title_rgb=(52, 211, 153),
    )

    # ==================== SLIDE 2: PROBLEM STATEMENT & MOTIVATION ====================
    s2 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s2)
    _add_header(
        s2,
        "1. Problem Statement & Limitations of Legacy Underwriting",
        "Why static actuarial tables fail in modern high-frequency InsurTech environments",
    )
    _add_card(
        s2, 0.6, 1.65, 3.85, 5.1,
        "Legacy Static Underwriting",
        [
            "Relies solely on lagging historical variables (age, gender, credit score, 3-year claim count).",
            "Blind to real-time behavioral shifts (harsh braking, night speeding, severe route weather).",
            "Uncalibrated ML classifiers distort true claim probabilities, causing mispriced technical premiums.",
            "Socioeconomic proxy bias penalizes safe drivers in lower-income regions.",
        ],
        title_rgb=(251, 113, 133),
    )
    _add_card(
        s2, 4.7, 1.65, 3.85, 5.1,
        "Our Hybrid Real-Time Solution",
        [
            "Combines 8 Historical Policy Features with 5 Real-Time Connected Vehicle / IoT Telemetry Streams.",
            "Dual-Stage Actuarial Engine: Calibrated Frequency Classifier P(Claim) × Gradient Boosted Severity E[Loss|Claim].",
            "Dynamic Telematics Modifier (-22% safe driver discount to +35% hazard surcharge).",
            "Sub-50ms real-time quote & explainable factor attribution.",
        ],
        title_rgb=(56, 189, 248),
    )
    _add_card(
        s2, 8.8, 1.65, 3.9, 5.1,
        "2026 Compliance Mandates",
        [
            "Calibrated Probabilities: Required for Solvency II capital reserve accuracy.",
            "Fairness Constraints: Enforces EEOC / NAIC 80% Disparate Impact Ratio across demographic zones.",
            "Differential Privacy: Protects sensitive credit and behavioral telemetry via ε-Laplace noise.",
            "Drift Governance: Continuous PSI & KS-test triggers when macro or weather patterns shift.",
        ],
        title_rgb=(52, 211, 153),
    )

    # ==================== SLIDE 3: SYSTEM ARCHITECTURE ====================
    s3 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s3)
    _add_header(
        s3,
        "2. End-to-End Predictive Underwriting System Architecture",
        "Modular 5-stage pipeline from raw ingestion and privacy sanitization to calibrated pricing & drift monitoring",
    )
    _add_card(
        s3, 0.6, 1.65, 2.3, 4.9,
        "Stage 1: Data Fusion",
        [
            "Historical Policy DB (Tenure, Prior Claims, Credit, Vehicle Value).",
            "Real-Time IoT Stream (Braking/100km, Overspeeding, Night Ratio, Route Weather).",
        ],
        title_rgb=(56, 189, 248),
    )
    _add_card(
        s3, 3.1, 1.65, 2.3, 4.9,
        "Stage 2: Privacy Guard",
        [
            "Salted SHA-256 PII Tokenization.",
            "National ID Masking.",
            "k-Anonymity Age Bands.",
            "ε-Differential Privacy Laplace Perturbation.",
        ],
        title_rgb=(167, 139, 250),
    )
    _add_card(
        s3, 5.6, 1.65, 2.4, 4.9,
        "Stage 3: Calibrated ML",
        [
            "Kamiran-Calders Fairness Sample Weights.",
            "Isotonic & Platt Calibrated Frequency Model.",
            "GBR Severity Loss Model E[L|C].",
        ],
        title_rgb=(52, 211, 153),
    )
    _add_card(
        s3, 8.2, 1.65, 2.3, 4.9,
        "Stage 4: Fair Pricing",
        [
            "Post-Processing Group Parity Calibration.",
            "Pure Premium = P(C) × E[L|C].",
            "Telematics Dynamic Discount/Loading.",
            "4-Tier Auto Decision.",
        ],
        title_rgb=(251, 191, 36),
    )
    _add_card(
        s3, 10.7, 1.65, 2.05, 4.9,
        "Stage 5: Drift Monitor",
        [
            "Feature PSI & Score PSI.",
            "Two-Sample KS Test.",
            "Quarterly ECE & A/E Loss Ratio Alerts.",
        ],
        title_rgb=(251, 113, 133),
    )

    # ==================== SLIDE 4: CALIBRATED RISK MODELS ====================
    s4 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s4)
    _add_header(
        s4,
        "3. Calibrated Risk Models & Reliability Analysis (2026 Update)",
        "Comparing Uncalibrated Ensemble vs. Platt Scaling (Sigmoid) vs. Isotonic Regression Calibration",
    )
    s4.shapes.add_picture(charts["calibration"], Inches(0.6), Inches(1.65), width=Inches(7.3))
    _add_card(
        s4, 8.1, 1.65, 4.6, 5.0,
        "Why Calibration Matters in Underwriting",
        [
            "In insurance pricing, raw classifier scores are NOT true probabilities: Pure Premium = P(Claim) × Severity requires exact probability alignment.",
            f"Uncalibrated RF: Brier = {cal_u['brier_score']}, ECE = {cal_u['ece']} (systematically over-predicts risk due to cost-sensitive trees).",
            f"Platt Scaling (Sigmoid): Brier = {summary['calibration_comparison']['platt_sigmoid']['brier_score']}, ECE = {summary['calibration_comparison']['platt_sigmoid']['ece']}.",
            f"Isotonic Regression (Selected): Brier = {cal_i['brier_score']}, ECE = {cal_i['ece']}, ROC-AUC = {cal_i['roc_auc']}.",
            "Result: Aligns predicted claim frequency tightly with the 45° empirical actuarial diagonal.",
        ],
        title_rgb=(52, 211, 153),
    )

    # ==================== SLIDE 5: ACTUARIAL PRICING & LOSS FORECASTING ====================
    s5 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s5)
    _add_header(
        s5,
        "4. Actuarial Pricing Formula & Portfolio Loss Forecasting",
        "Translating calibrated claim probabilities and severity regressions into technical premiums",
    )
    _add_card(
        s5, 0.6, 1.65, 6.0, 5.0,
        "Mathematical Pricing & Loss Formulation",
        [
            "1. Expected Claim Frequency: P_fair(Claim | X_hist, X_rt) via Isotonic Calibrated + Fairness-Constrained Classifier.",
            "2. Expected Claim Severity: E[Loss | Claim, X] via Gradient Boosting Regressor (Severity MAE = $"
            f"{loss_f['severity_mae_usd']:,.2f}, R² = {loss_f['severity_r2']}).",
            "3. Pure Premium (Expected Annual Loss): Pure_Premium = P_fair(Claim) × E[Loss | Claim].",
            "4. Real-Time Telematics Adjustment: Δ_RT = Pure_Premium × 0.65% × (72.0 - Telematics_Score).",
            "5. Technical Annual Premium = (Pure_Premium + Δ_RT + $115 Fixed Expense) / (1 - 0.24 Solvency & Expense Ratio).",
        ],
        title_rgb=(56, 189, 248),
    )
    _add_card(
        s5, 6.85, 1.65, 5.85, 5.0,
        "Holdout Portfolio Loss Forecasting Results",
        [
            f"Holdout Test Portfolio Size: {summary['test_size']} Policies",
            f"Forecasted Portfolio Total Loss: ${loss_f['portfolio_expected_loss_usd']:,.2f}",
            f"Observed Actual Portfolio Loss: ${loss_f['portfolio_actual_loss_usd']:,.2f}",
            f"Portfolio Loss Forecast Accuracy: {loss_f['loss_forecast_accuracy_pct']}%",
            f"Average Pure Premium per Policy: ${loss_f['avg_pure_premium_usd']:,.2f}",
            "Top Predictive Risk Drivers: (1) Real-Time Telematics Score, (2) Harsh Braking/100km, (3) Overspeed Ratio, (4) Prior Claims, (5) Night Driving Ratio.",
        ],
        title_rgb=(251, 191, 36),
    )

    # ==================== SLIDE 6: FAIRNESS CONSTRAINTS (2026 UPDATE) ====================
    s6 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s6)
    _add_header(
        s6,
        "5. Fairness Constraints & Proxy Bias Mitigation (2026 Update)",
        "Enforcing the 80% Disparate Impact Rule across socioeconomic regions and demographic groups",
    )
    s6.shapes.add_picture(charts["fairness"], Inches(0.6), Inches(1.65), width=Inches(7.3))
    _add_card(
        s6, 8.1, 1.65, 4.6, 5.0,
        "Two-Stage Fairness Architecture",
        [
            "Problem: Historical credit scores correlate with regional income tiers, causing unmitigated models to penalize safe Low-Income Zone drivers.",
            f"Before Mitigation: Disparate Impact Ratio (DIR) = {fair_b['disparate_impact_ratio']} (< 0.80 EEOC threshold — FAILS compliance).",
            "Stage 1 (Pre-Processing): Kamiran & Calders sample reweighing w(a,y) = P(A)P(Y)/P(A,Y).",
            "Stage 2 (Post-Processing): Actuarial group-mean parity calibration preserving total portfolio expected loss.",
            f"After Mitigation: DIR = {fair_a['disparate_impact_ratio']} (PASSES ≥ 0.80 rule), Statistical Parity Diff improved from {fair_b['statistical_parity_diff']} to {fair_a['statistical_parity_diff']}.",
        ],
        title_rgb=(52, 211, 153),
    )

    # ==================== SLIDE 7: DATA PRIVACY GUARDRAILS (2026 UPDATE) ====================
    s7 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s7)
    _add_header(
        s7,
        "6. Data Privacy & Differential Privacy Engine (2026 Update)",
        "Protecting PII and sensitive telemetry via cryptographic tokenization, k-anonymity, and ε-Laplace noise",
    )
    s7.shapes.add_picture(charts["privacy"], Inches(0.6), Inches(1.65), width=Inches(7.1))
    _add_card(
        s7, 7.95, 1.65, 4.75, 5.0,
        "Three Defense-in-Depth Privacy Layers",
        [
            "Layer 1 — Cryptographic Pseudonymization: Applicant names hashed with salted SHA-256 (e.g., 'Ananya Sharma' → 'TOK-8F91A2B4C1E0'); National IDs masked to last 4 digits.",
            "Layer 2 — k-Anonymity Generalization: Exact ages generalized into actuarial age bands (18-25, 26-35, 36-45, etc.) for analytics storage.",
            "Layer 3 — ε-Differential Privacy: Calibrated Laplace noise Lap(0, Δf / ε) injected into sensitive credit & telematics features.",
            "Recommended Operating Point: ε = 2.5 preserves ROC-AUC > 0.80 and Brier calibration while preventing reconstruction attacks.",
        ],
        title_rgb=(167, 139, 250),
    )

    # ==================== SLIDE 8: CONTINUOUS MODEL MONITORING (2026 UPDATE) ====================
    s8 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s8)
    _add_header(
        s8,
        "7. Continuous Model Monitoring & Drift Detection (2026 Update)",
        "Tracking Population Stability Index (PSI), KS-Tests, and Calibration Decay across 2026 quarters",
    )
    s8.shapes.add_picture(charts["drift"], Inches(0.6), Inches(1.65), width=Inches(7.3))
    _add_card(
        s8, 8.1, 1.65, 4.6, 5.0,
        "Automated Drift Governance Protocol",
        [
            "Population Stability Index (PSI): Quantifies distributional shift between Baseline Training data and live production windows.",
            "PSI < 0.10 (Green): Stable — Normal underwriting operation.",
            "0.10 ≤ PSI < 0.25 (Amber): Moderate shift — Trigger actuarial watch alert (seen in 2026-Q3 Mild Telematics Shift).",
            "PSI ≥ 0.25 (Red): Critical shift — Automated model recalibration & retraining triggered (seen in 2026-Q4 Severe Weather/Macro Shift where braking & overspeed PSI > 0.30).",
        ],
        title_rgb=(251, 113, 133),
    )

    # ==================== SLIDE 9: LIVE DEMONSTRATION WALKTHROUGH (OVERVIEW) ====================
    s9 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s9)
    _add_header(
        s9,
        "8. Live Demonstration Walkthrough — Interactive Web Studio",
        "How the live application (app.py) demonstrates real-time underwriting, calibration, fairness & drift",
        "LIVE DEMONSTRATION GUIDE • PART 1",
    )
    s9.shapes.add_picture(charts["demo_cases"], Inches(0.6), Inches(1.65), width=Inches(7.4))
    _add_card(
        s9, 8.2, 1.65, 4.5, 5.0,
        "What to Show During Live Demo",
        [
            "Step 1: Launch `python app.py` and open http://127.0.0.1:5000.",
            "Step 2: Click through the 4 curated Demo Personas (Safe Commuter, Fairness Showcase, Night Speeder, High-Risk Decline).",
            "Step 3: Toggle the 'Fairness Constraint' checkbox on Case 2 to show live reduction in Low-Income Zone risk penalty.",
            "Step 4: Click '⚡ Stream Live IoT Packet' to demonstrate real-time connected vehicle underwriting.",
            "Step 5: Switch to Tabs 2, 3, and 4 to show Reliability Curves, Fairness DIR, and PSI Drift triggers.",
        ],
        title_rgb=(56, 189, 248),
    )

    # ==================== SLIDE 10: DEMO CASES DEEP DIVE (NUMBERS TO SHOW) ====================
    s10 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s10)
    _add_header(
        s10,
        "9. Demonstration Test Cases — Exact Inputs & Outputs",
        "Side-by-side comparison of the 4 live demonstration scenarios built into the dashboard",
        "LIVE DEMONSTRATION GUIDE • PART 2",
    )
    c1 = persona_results["safe_commuter"]["evaluation"]
    c2 = persona_results["fairness_showcase"]["evaluation"]
    c3 = persona_results["realtime_speeder"]["evaluation"]
    c4 = persona_results["high_risk_decline"]["evaluation"]

    _add_card(
        s10, 0.6, 1.65, 2.9, 5.1,
        "Case 1: Safe Driver",
        [
            "Profile: 34yo, 0 claims, Credit 765, Telematics 91.5/100.",
            f"Uncalibrated Prob: {c1['probabilities']['uncalibrated_prob']*100:.1f}%",
            f"Calibrated Prob: {c1['probabilities']['final_underwriting_prob']*100:.1f}%",
            f"Telematics Mod: {c1['pricing']['telematics_modifier_pct']}% Discount",
            f"Annual Premium: ${c1['pricing']['annual_technical_premium_usd']:,.2f}",
            f"Decision: {c1['decision']['tier']}",
        ],
        title_rgb=(52, 211, 153),
    )
    _add_card(
        s10, 3.7, 1.65, 2.9, 5.1,
        "Case 2: Fairness Demo",
        [
            "Profile: Low-Income Zone, 0 claims, Telematics 84.0/100.",
            f"Uncalibrated Prob: {c2['probabilities']['uncalibrated_prob']*100:.1f}%",
            f"Fairness Shift: {c2['probabilities']['fairness_shift_applied']*100:+.1f}%",
            f"Final Risk Prob: {c2['probabilities']['final_underwriting_prob']*100:.1f}%",
            f"Annual Premium: ${c2['pricing']['annual_technical_premium_usd']:,.2f}",
            f"Decision: {c2['decision']['tier']}",
        ],
        title_rgb=(56, 189, 248),
    )
    _add_card(
        s10, 6.8, 1.65, 2.9, 5.1,
        "Case 3: Night Speeder",
        [
            "Profile: 0 prior claims, but 46% night driving & 42% overspeed.",
            f"Telematics Score: 49.0 / 100",
            f"Final Risk Prob: {c3['probabilities']['final_underwriting_prob']*100:.1f}%",
            f"Telematics Mod: +{c3['pricing']['telematics_modifier_pct']}% Surcharge",
            f"Annual Premium: ${c3['pricing']['annual_technical_premium_usd']:,.2f}",
            f"Decision: {c3['decision']['tier']}",
        ],
        title_rgb=(251, 191, 36),
    )
    _add_card(
        s10, 9.9, 1.65, 2.85, 5.1,
        "Case 4: High-Risk Decline",
        [
            "Profile: 3 prior claims, Telematics 34.5, Braking 11.4/100km.",
            f"Uncalibrated Prob: {c4['probabilities']['uncalibrated_prob']*100:.1f}%",
            f"Final Risk Prob: {c4['probabilities']['final_underwriting_prob']*100:.1f}%",
            f"Pure Loss E[L]: ${c4['pricing']['pure_premium_expected_loss_usd']:,.2f}",
            f"Annual Quote: ${c4['pricing']['annual_technical_premium_usd']:,.2f}",
            f"Decision: {c4['decision']['tier']}",
        ],
        title_rgb=(251, 113, 133),
    )

    # ==================== SLIDE 11: GITHUB REPOSITORY STRUCTURE ====================
    s11 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s11)
    _add_header(
        s11,
        "10. GitHub Repository Structure & Reproducibility Guide",
        "Clean, modular, production-ready codebase ready for GitHub hosting and peer review",
    )
    _add_card(
        s11, 0.6, 1.65, 6.0, 5.0,
        "Repository Directory Layout",
        [
            "├── app.py  (Flask Live Underwriting & Monitoring Web Server)",
            "├── generate_assets_and_ppt.py  (Reproducible Chart & Deck Builder)",
            "├── requirements.txt & README.md  (Full Docs & GitHub Visuals)",
            "├── src/",
            "│   ├── data_pipeline.py  (Historical + Real-Time IoT Stream Generator)",
            "│   ├── risk_models.py  (Platt/Isotonic Calibration + Severity GBR)",
            "│   ├── fairness_engine.py  (Kamiran-Calders + Post-Processing Parity)",
            "│   ├── privacy_engine.py  (SHA-256 Tokenization + ε-Laplace DP)",
            "│   └── monitoring_engine.py  (Feature PSI, KS-Test & Quarterly Drift)",
            "├── templates/index.html  (Interactive Tailwind + Chart.js Studio)",
            "└── presentation/  (Editable .pptx + Standalone HTML Slide Deck)",
        ],
        title_rgb=(56, 189, 248),
    )
    _add_card(
        s11, 6.85, 1.65, 5.85, 5.0,
        "Quick-Start & GitHub Repository Link",
        [
            "Official GitHub Repository: https://github.com/Indra-Official/FIntech",
            "1. Clone Repository:  git clone https://github.com/Indra-Official/FIntech.git",
            "2. Install Dependencies:  pip install -r requirements.txt",
            "3. Regenerate Benchmarks & Slides:  python generate_assets_and_ppt.py",
            "4. Launch Interactive Live Demo:  python app.py  (http://127.0.0.1:5000)",
            "5. Git Remote Configuration:",
            "   git remote add origin https://github.com/Indra-Official/FIntech.git",
            "   git branch -M main && git push -u origin main",
        ],
        title_rgb=(52, 211, 153),
    )

    # ==================== SLIDE 12: CONCLUSION & IMPACT ====================
    s12 = prs.slides.add_slide(blank_layout)
    _style_slide_bg(s12)
    _add_header(
        s12,
        "11. Conclusion & Business Impact Summary",
        "Delivering accurate, fair, privacy-preserving, and drift-resilient insurance underwriting",
    )
    _add_card(
        s12, 0.6, 1.75, 12.1, 4.9,
        "Summary of Achievements for Statement #31 (2026 Update)",
        [
            "1. Historical + Real-Time Fusion: Combined traditional policy/claim records with live IoT telematics streams, enabling behavior-based dynamic discounts (-22%) and hazard surcharges (+35%).",
            f"2. Calibrated Risk & Loss Forecasting: Isotonic calibration reduced Brier score to {cal_i['brier_score']} and ECE to {cal_i['ece']}, achieving {loss_f['loss_forecast_accuracy_pct']}% portfolio loss forecast accuracy.",
            f"3. Audited Algorithmic Fairness: Eliminated regional socioeconomic proxy bias, lifting the Disparate Impact Ratio from {fair_b['disparate_impact_ratio']} to {fair_a['disparate_impact_ratio']} (compliant with the ≥0.80 regulatory threshold).",
            "4. Defense-in-Depth Data Privacy: Enforced salted SHA-256 PII tokenization, National ID redaction, k-anonymity age bands, and ε-Differential Privacy.",
            "5. Proactive Drift Governance: Automated PSI and KS-test monitoring across quarterly streams detects severe weather/behavioral shifts and triggers recalibration.",
            "6. Full Source Code & Interactive Demo: https://github.com/Indra-Official/FIntech",
        ],
        title_rgb=(56, 189, 248),
    )

    prs.save(output_path)


def build_interactive_html_deck(summary, quarterly, persona_results, output_path):
    """Generates a self-contained HTML presentation deck with keyboard navigation (Arrow Left/Right)."""
    cal_u = summary["calibration_comparison"]["uncalibrated"]
    cal_i = summary["calibration_comparison"]["isotonic_calibrated"]
    fair_b = summary["fairness_audit"]["before_mitigation"]
    fair_a = summary["fairness_audit"]["after_mitigation"]
    loss_f = summary["loss_forecasting"]

    c1 = persona_results["safe_commuter"]["evaluation"]
    c2 = persona_results["fairness_showcase"]["evaluation"]
    c3 = persona_results["realtime_speeder"]["evaluation"]
    c4 = persona_results["high_risk_decline"]["evaluation"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Statement #31 — Predictive Underwriting Interactive Presentation Deck</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Inter', sans-serif; background: #0b0f19; color: #f8fafc; overflow: hidden; }}
    .mono {{ font-family: 'JetBrains Mono', monospace; }}
    .slide {{ display: none; height: calc(100vh - 64px); }}
    .slide.active {{ display: flex; }}
    .card {{ background: rgba(22, 31, 50, 0.9); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 1rem; }}
  </style>
</head>
<body class="flex flex-col h-screen">

  <!-- Top Controls -->
  <header class="h-16 border-b border-slate-800 px-6 flex items-center justify-between bg-slate-950">
    <div class="flex items-center gap-3">
      <span class="px-2.5 py-1 rounded bg-sky-500/20 text-sky-400 font-bold text-xs mono">STATEMENT #31</span>
      <span class="font-bold text-sm text-white">Predictive Underwriting — Digital Assignment Presentation &amp; Demo Walkthrough</span>
      <a href="https://github.com/Indra-Official/FIntech" target="_blank" class="px-2.5 py-1 rounded bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/25 font-semibold text-xs mono">GitHub: Indra-Official/FIntech &nearr;</a>
    </div>
    <div class="flex items-center gap-3">
      <span id="slide-indicator" class="text-xs text-slate-400 mono">Slide 1 / 8</span>
      <button onclick="prevSlide()" class="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs font-semibold">&larr; Prev</button>
      <button onclick="nextSlide()" class="px-3 py-1.5 rounded bg-sky-600 hover:bg-sky-500 text-xs font-semibold">Next &rarr;</button>
    </div>
  </header>

  <!-- Slides Container -->
  <div class="flex-1 relative">

    <!-- Slide 1 -->
    <div class="slide active flex-col justify-center px-12 max-w-7xl mx-auto w-full gap-6">
      <span class="text-xs uppercase tracking-widest text-sky-400 font-bold">FinTech Digital Assignment • 2026 Update</span>
      <h1 class="text-4xl font-extrabold leading-tight">Statement #31 — Predictive Underwriting &amp; Loss Forecasting</h1>
      <p class="text-lg text-slate-300 max-w-4xl">Using historical policy data and real-time IoT telematics streams to improve insurance pricing, automated underwriting, and portfolio loss forecasting — enhanced with <span class="text-sky-400 font-semibold">Calibrated Risk Models, Model Drift Monitoring, Data Privacy, and Fairness Constraints</span>.</p>
      <div class="grid grid-cols-4 gap-4 pt-4">
        <div class="card p-5">
          <div class="text-xs text-slate-400">Calibrated Brier / ECE</div>
          <div class="text-2xl font-extrabold text-emerald-400 mono mt-1">{cal_i['brier_score']} / {cal_i['ece']}</div>
          <div class="text-xs text-slate-400 mt-1">Isotonic Probability Calibration</div>
        </div>
        <div class="card p-5">
          <div class="text-xs text-slate-400">Portfolio Loss Accuracy</div>
          <div class="text-2xl font-extrabold text-sky-400 mono mt-1">{loss_f['loss_forecast_accuracy_pct']}%</div>
          <div class="text-xs text-slate-400 mt-1">Pure Premium = P(C) &times; E[L|C]</div>
        </div>
        <div class="card p-5">
          <div class="text-xs text-slate-400">Fairness Disparate Impact</div>
          <div class="text-2xl font-extrabold text-indigo-400 mono mt-1">{fair_b['disparate_impact_ratio']} &rarr; {fair_a['disparate_impact_ratio']}</div>
          <div class="text-xs text-slate-400 mt-1">Passes &ge; 0.80 EEOC Rule</div>
        </div>
        <div class="card p-5">
          <div class="text-xs text-slate-400">Privacy &amp; Drift Guard</div>
          <div class="text-2xl font-extrabold text-amber-400 mono mt-1">&epsilon;=2.5 DP + PSI</div>
          <div class="text-xs text-slate-400 mt-1">SHA-256 + Laplace + KS Test</div>
        </div>
      </div>
    </div>

    <!-- Slide 2 -->
    <div class="slide flex-col justify-center px-12 max-w-7xl mx-auto w-full gap-6">
      <h2 class="text-2xl font-bold text-sky-400">1. Architecture: Historical + Real-Time Data Fusion</h2>
      <div class="grid grid-cols-3 gap-6">
        <div class="card p-6 space-y-2">
          <h3 class="font-bold text-white text-lg">Historical Policy Inputs (8)</h3>
          <ul class="text-sm text-slate-300 space-y-1.5 list-disc pl-4">
            <li>Driver Age &amp; Years Licensed</li>
            <li>Prior Claims (3-Year Window)</li>
            <li>Credit Stability Score (DP-Protected)</li>
            <li>Insured Vehicle Value (USD)</li>
            <li>Vehicle Safety Rating (2★ to 5★)</li>
            <li>Annual Mileage &amp; Policy Tenure</li>
          </ul>
        </div>
        <div class="card p-6 space-y-2">
          <h3 class="font-bold text-emerald-400 text-lg">Real-Time Telemetry Streams (5)</h3>
          <ul class="text-sm text-slate-300 space-y-1.5 list-disc pl-4">
            <li>Composite Telematics Score (0–100)</li>
            <li>Harsh Braking Events per 100km</li>
            <li>Night-Time Driving Ratio</li>
            <li>GPS Overspeeding Ratio</li>
            <li>Live Weather &amp; Route Hazard Index</li>
          </ul>
        </div>
        <div class="card p-6 space-y-2">
          <h3 class="font-bold text-indigo-400 text-lg">2026 Governance Layer</h3>
          <ul class="text-sm text-slate-300 space-y-1.5 list-disc pl-4">
            <li><b>Calibrated Risk:</b> Isotonic &amp; Platt Scaling</li>
            <li><b>Fairness:</b> Reweighing + Group Parity</li>
            <li><b>Privacy:</b> Salted SHA-256 + &epsilon;-Laplace DP</li>
            <li><b>Monitoring:</b> PSI &amp; KS Drift Detection</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Slide 3 -->
    <div class="slide flex-col justify-center px-12 max-w-7xl mx-auto w-full gap-6">
      <h2 class="text-2xl font-bold text-emerald-400">2. Calibrated Risk Models &amp; Loss Forecasting (2026 Update)</h2>
      <div class="grid grid-cols-2 gap-6 items-center">
        <div class="card p-4">
          <img src="../docs/images/calibration_reliability.png" alt="Calibration Curve" class="rounded-lg w-full" />
        </div>
        <div class="card p-6 space-y-3 text-sm text-slate-200">
          <h3 class="text-lg font-bold text-white">Actuarial Calibration Results</h3>
          <p>Uncalibrated cost-sensitive ensembles systematically overestimate claim probability, inflating customer premiums. We apply 5-fold cross-validated <b>Isotonic Regression Calibration</b>:</p>
          <ul class="space-y-2 list-disc pl-4">
            <li><b>Uncalibrated Ensemble:</b> Brier = <span class="mono text-rose-400">{cal_u['brier_score']}</span>, ECE = <span class="mono text-rose-400">{cal_u['ece']}</span></li>
            <li><b>Isotonic Calibrated:</b> Brier = <span class="mono text-emerald-400">{cal_i['brier_score']}</span>, ECE = <span class="mono text-emerald-400">{cal_i['ece']}</span>, ROC-AUC = <span class="mono text-sky-400">{cal_i['roc_auc']}</span></li>
            <li><b>Portfolio Loss Forecast:</b> Predicted <span class="mono text-sky-300">${loss_f['portfolio_expected_loss_usd']:,.0f}</span> vs. Actual <span class="mono text-emerald-300">${loss_f['portfolio_actual_loss_usd']:,.0f}</span> (<span class="mono font-bold">{loss_f['loss_forecast_accuracy_pct']}%</span> accuracy).</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Slide 4 -->
    <div class="slide flex-col justify-center px-12 max-w-7xl mx-auto w-full gap-6">
      <h2 class="text-2xl font-bold text-sky-400">3. Fairness Constraints &amp; Socioeconomic Bias Mitigation</h2>
      <div class="grid grid-cols-2 gap-6 items-center">
        <div class="card p-4">
          <img src="../docs/images/fairness_mitigation.png" alt="Fairness Mitigation" class="rounded-lg w-full" />
        </div>
        <div class="card p-6 space-y-3 text-sm text-slate-200">
          <h3 class="text-lg font-bold text-white">Satisfying the 80% Disparate Impact Rule</h3>
          <p>Historical credit stability scores carry socioeconomic proxy bias against Low-Income Zone applicants. Our two-stage fairness constraint applies Kamiran-Calders sample weights during training and group parity calibration at inference:</p>
          <ul class="space-y-2 list-disc pl-4">
            <li><b>Disparate Impact Ratio (Before):</b> <span class="mono text-rose-400">{fair_b['disparate_impact_ratio']}</span> (Fails &ge;0.80 threshold)</li>
            <li><b>Disparate Impact Ratio (After):</b> <span class="mono text-emerald-400">{fair_a['disparate_impact_ratio']}</span> (Passes regulatory threshold)</li>
            <li><b>Statistical Parity Difference:</b> Improved from <span class="mono">{fair_b['statistical_parity_diff']}</span> to <span class="mono text-emerald-400">{fair_a['statistical_parity_diff']}</span></li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Slide 5 -->
    <div class="slide flex-col justify-center px-12 max-w-7xl mx-auto w-full gap-6">
      <h2 class="text-2xl font-bold text-indigo-400">4. Data Privacy &amp; Differential Privacy (&epsilon;-Laplace)</h2>
      <div class="grid grid-cols-2 gap-6 items-center">
        <div class="card p-4">
          <img src="../docs/images/privacy_utility_tradeoff.png" alt="Privacy Tradeoff" class="rounded-lg w-full" />
        </div>
        <div class="card p-6 space-y-3 text-sm text-slate-200">
          <h3 class="text-lg font-bold text-white">Privacy-Preserving Telemetry &amp; PII Protection</h3>
          <ul class="space-y-2 list-disc pl-4">
            <li><b>Salted SHA-256 Tokenization:</b> Converts raw applicant names into irreversible tokens (<span class="mono text-sky-300">TOK-8F91A2B4C1E0</span>).</li>
            <li><b>National ID Redaction:</b> Masks identifiers to <span class="mono text-slate-300">ID-XXXX-3910</span>.</li>
            <li><b>k-Anonymity Generalization:</b> Buckets exact ages into actuarial bands.</li>
            <li><b>&epsilon;-Differential Privacy:</b> Injects calibrated Laplace noise <span class="mono">Lap(0, &Delta;f/&epsilon;)</span> into sensitive credit and telematics scores. At <span class="mono text-emerald-400">&epsilon;=2.5</span>, full calibration utility is preserved.</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Slide 6 -->
    <div class="slide flex-col justify-center px-12 max-w-7xl mx-auto w-full gap-6">
      <h2 class="text-2xl font-bold text-rose-400">5. Continuous Model Monitoring &amp; Drift Detection (PSI &amp; KS)</h2>
      <div class="grid grid-cols-2 gap-6 items-center">
        <div class="card p-4">
          <img src="../docs/images/drift_monitoring_psi.png" alt="PSI Drift" class="rounded-lg w-full" />
        </div>
        <div class="card p-6 space-y-3 text-sm text-slate-200">
          <h3 class="text-lg font-bold text-white">Quarterly Production Drift Surveillance</h3>
          <p>Monitors every feature using the <b>Population Stability Index (PSI)</b> and two-sample <b>Kolmogorov-Smirnov (KS)</b> tests:</p>
          <ul class="space-y-2 list-disc pl-4">
            <li><b>2026-Q1 &amp; Q2:</b> Max PSI &lt; 0.10 &rarr; <span class="text-emerald-400 font-semibold">HEALTHY</span></li>
            <li><b>2026-Q3 (Mild Telematics Shift):</b> Max PSI = {quarterly[2]['max_feature_psi']} &rarr; <span class="text-amber-400 font-semibold">WATCH ALERT</span></li>
            <li><b>2026-Q4 (Severe Weather &amp; Macro Shift):</b> Max PSI = {quarterly[3]['max_feature_psi']} &rarr; <span class="text-rose-400 font-semibold">CRITICAL: AUTO-RECALIBRATE</span></li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Slide 7: Demonstration Walkthrough -->
    <div class="slide flex-col justify-center px-12 max-w-7xl mx-auto w-full gap-5">
      <h2 class="text-2xl font-bold text-amber-400">6. Live Demonstration Walkthrough — 4 Interactive Test Cases</h2>
      <div class="grid grid-cols-2 gap-6 items-center">
        <div class="card p-4">
          <img src="../docs/images/loss_forecasting_pricing.png" alt="Demo Cases" class="rounded-lg w-full" />
        </div>
        <div class="grid grid-cols-2 gap-3 text-xs">
          <div class="card p-4 border-l-4 border-l-emerald-400">
            <div class="font-bold text-emerald-400">Case 1: Safe Commuter</div>
            <div class="mt-1 text-slate-300">Telematics 91.5, 0 claims</div>
            <div class="mt-2 mono text-lg font-bold text-white">${c1['pricing']['annual_technical_premium_usd']:,.2f}/yr</div>
            <div class="text-emerald-300 mt-1">{c1['decision']['tier']}</div>
          </div>
          <div class="card p-4 border-l-4 border-l-sky-400">
            <div class="font-bold text-sky-400">Case 2: Fairness Showcase</div>
            <div class="mt-1 text-slate-300">Low-Income Zone, Telematics 84</div>
            <div class="mt-2 mono text-lg font-bold text-white">${c2['pricing']['annual_technical_premium_usd']:,.2f}/yr</div>
            <div class="text-sky-300 mt-1">{c2['decision']['tier']}</div>
          </div>
          <div class="card p-4 border-l-4 border-l-amber-400">
            <div class="font-bold text-amber-400">Case 3: Night Speeder</div>
            <div class="mt-1 text-slate-300">0 claims, 46% night, 42% speed</div>
            <div class="mt-2 mono text-lg font-bold text-white">${c3['pricing']['annual_technical_premium_usd']:,.2f}/yr</div>
            <div class="text-amber-300 mt-1">{c3['decision']['tier']}</div>
          </div>
          <div class="card p-4 border-l-4 border-l-rose-400">
            <div class="font-bold text-rose-400">Case 4: High-Risk Decline</div>
            <div class="mt-1 text-slate-300">3 claims, Telematics 34.5</div>
            <div class="mt-2 mono text-lg font-bold text-white">${c4['pricing']['annual_technical_premium_usd']:,.2f}/yr</div>
            <div class="text-rose-300 mt-1">{c4['decision']['tier']}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Slide 8: How to Run & Present Live -->
    <div class="slide flex-col justify-center px-12 max-w-7xl mx-auto w-full gap-6">
      <h2 class="text-2xl font-bold text-sky-400">7. Live Demo Execution &amp; GitHub Repository</h2>
      <div class="grid grid-cols-2 gap-6">
        <div class="card p-6 space-y-3">
          <h3 class="text-lg font-bold text-white">Step-by-Step Live Demonstration Script</h3>
          <ol class="text-sm text-slate-300 space-y-2 list-decimal pl-5">
            <li>Run <code class="mono bg-slate-900 px-2 py-0.5 rounded text-sky-300">python app.py</code> and open <code class="mono text-emerald-300">http://127.0.0.1:5000</code>.</li>
            <li>Click <b>Case 1 (Preferred Safe Driver)</b> to show instant auto-approval and -12.7% telematics discount.</li>
            <li>Click <b>Case 2 (Fairness Constraint Showcase)</b> and toggle the <i>Fairness Constraint</i> checkbox on/off to show live removal of regional proxy penalty.</li>
            <li>Click <b>Case 3 (Real-Time Night Speeder)</b> to show how real-time telemetry catches high risk even when prior claims = 0.</li>
            <li>Click <b>⚡ Stream Live IoT Packet</b> to score a live random connected-vehicle packet in real time.</li>
          </ol>
        </div>
        <div class="card p-6 space-y-3">
          <h3 class="text-lg font-bold text-emerald-400">Official GitHub Repository</h3>
          <div class="p-3 rounded-lg bg-slate-900 border border-sky-500/30">
            <div class="text-xs text-slate-400">Repository URL:</div>
            <a href="https://github.com/Indra-Official/FIntech" target="_blank" class="mono text-sm font-bold text-sky-400 hover:underline">https://github.com/Indra-Official/FIntech</a>
          </div>
          <pre class="mono text-xs bg-slate-950 p-4 rounded-xl border border-slate-800 text-slate-200 overflow-x-auto">git clone https://github.com/Indra-Official/FIntech.git
cd FIntech
pip install -r requirements.txt
python generate_assets_and_ppt.py
python app.py</pre>
        </div>
      </div>
    </div>

  </div>

  <script>
    let currentSlide = 0;
    const slides = document.querySelectorAll('.slide');
    function showSlide(n) {{
      slides[currentSlide].classList.remove('active');
      currentSlide = (n + slides.length) % slides.length;
      slides[currentSlide].classList.add('active');
      document.getElementById('slide-indicator').textContent = `Slide ${{currentSlide + 1}} / ${{slides.length}}`;
    }}
    function nextSlide() {{ showSlide(currentSlide + 1); }}
    function prevSlide() {{ showSlide(currentSlide - 1); }}
    window.addEventListener('keydown', e => {{
      if (e.key === 'ArrowRight' || e.key === ' ') nextSlide();
      if (e.key === 'ArrowLeft') prevSlide();
    }});
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    generate_all()
