"""
UI Components and Glassmorphic Theme Engine for EMIPredict AI.

Provides responsive styling, interactive cards, status badges, risk gauges,
and compliance warning callouts following modern financial design aesthetics.
"""

from typing import Dict, Optional
import streamlit as st


def inject_custom_css():
    """Injects high-end fintech dark/glassmorphic styling with Google Fonts."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, .brand-title {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* Primary brand gradient text */
    .brand-gradient {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    /* Metric card */
    .metric-container {
        display: flex;
        flex-direction: column;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        min-width: 140px;
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-subtitle {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Status badge pill */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-eligible {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .badge-high-risk {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
    }
    .badge-not-eligible {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }

    /* Warning callout */
    .alert-box {
        border-radius: 12px;
        padding: 16px 20px;
        margin: 14px 0;
        font-size: 0.90rem;
        line-height: 1.5;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .alert-warning {
        background: rgba(245, 158, 11, 0.16);
        border-left: 5px solid #F59E0B;
        color: #FCD34D;
    }
    .alert-info {
        background: rgba(56, 189, 248, 0.16);
        border-left: 5px solid #38BDF8;
        color: #BAE6FD;
    }
    .alert-danger {
        background: rgba(239, 68, 68, 0.18);
        border-left: 5px solid #EF4444;
        color: #FCA5A5;
    }

    /* Probability bar split */
    .prob-bar-wrapper {
        width: 100%;
        background: #1E293B;
        border-radius: 9999px;
        height: 14px;
        overflow: hidden;
        display: flex;
        margin: 10px 0;
    }
    .prob-seg-elig {
        background: #10B981;
        height: 100%;
        transition: width 0.4s ease;
    }
    .prob-seg-risk {
        background: #F59E0B;
        height: 100%;
        transition: width 0.4s ease;
    }
    .prob-seg-not {
        background: #EF4444;
        height: 100%;
        transition: width 0.4s ease;
    }

    /* Quick simulation slider */
    .sim-box {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def status_badge_html(status: str) -> str:
    """Returns HTML for a glowing pill badge matching the predicted risk class."""
    status_clean = str(status).strip()
    if status_clean == "Eligible":
        return '<span class="badge badge-eligible">✓ Eligible</span>'
    elif status_clean == "High_Risk":
        return '<span class="badge badge-high-risk">⚠ High Risk (Borderline)</span>'
    elif status_clean == "Not_Eligible":
        return '<span class="badge badge-not-eligible">✕ Not Eligible</span>'
    return f'<span class="badge">{status}</span>'


def metric_card_html(title: str, value: str, subtitle: str = "") -> str:
    """Renders a modern metric card in HTML."""
    sub_html = f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="metric-container">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """


def render_probability_bars(probabilities: Dict[str, float]):
    """Renders visual multi-class probability progress bars and statistics."""
    p_elig = probabilities.get("Eligible", 0.0) * 100.0
    p_risk = probabilities.get("High_Risk", 0.0) * 100.0
    p_not = probabilities.get("Not_Eligible", 0.0) * 100.0

    html = f"""
    <div style="margin: 12px 0;">
        <div class="prob-bar-wrapper">
            <div class="prob-seg-elig" style="width: {p_elig}%;" title="Eligible: {p_elig:.1f}%"></div>
            <div class="prob-seg-risk" style="width: {p_risk}%;" title="High Risk: {p_risk:.1f}%"></div>
            <div class="prob-seg-not" style="width: {p_not}%;" title="Not Eligible: {p_not:.1f}%"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1;">
            <span><span style="color:#34D399; font-weight:600;">●</span> Eligible: <strong>{p_elig:.1f}%</strong></span>
            <span><span style="color:#FBBF24; font-weight:600;">●</span> High Risk: <strong>{p_risk:.1f}%</strong></span>
            <span><span style="color:#F87171; font-weight:600;">●</span> Not Eligible: <strong>{p_not:.1f}%</strong></span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def high_risk_warning_callout():
    """Renders mandatory disclaimer when High_Risk is predicted."""
    html = """
    <div class="alert-box alert-warning">
        <strong>⚠️ CRITICAL UNDERWRITING WARNING — BORDERLINE AFFORDABILITY FLAG:</strong><br>
        This applicant was classified as <strong>High Risk</strong>. In our held-out test evaluation on 80,960 records,
        the <code>High_Risk</code> category demonstrates <strong>20.08% precision</strong> and <strong>80.93% recall</strong>.
        <br><br>
        <strong>What this means:</strong> The model intentionally casts a wide net to capture ~81% of borderline debt-service applicants
        (where implied monthly obligation is ~78% to 97% of maximum capacity). <strong>This classification must NOT be used
        for automated credit decline.</strong> Mandatory protocol requires secondary manual income verification, debt-obligation audits,
        or alternative collateral assessment.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def floor_policy_callout(raw_val: float, policy_val: float):
    """Explains dataset-derived policy floor adjustment when raw capacity is below ₹500."""
    html = f"""
    <div class="alert-box alert-info">
        <strong>ℹ️ OPTIONAL DATASET-DERIVED POLICY FLOOR APPLIED:</strong><br>
        The unconstrained mathematical model estimated monthly debt capacity at <strong>₹{raw_val:,.2f}</strong>
        (indicating existing expenses and debts fully consume or exceed declared salary).
        <br><br>
        <strong>Policy Note:</strong> In accordance with retail underwriting design and the observed distribution floor
        of the training dataset, capacity is bounded by an optional baseline policy limit of <strong>₹{policy_val:,.2f} / month</strong>.
        This ₹500 threshold is an empirical dataset-derived heuristic and operational policy choice, <em>not a statutory or regulatory banking mandate</em>.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def synthetic_disclaimer_callout():
    """Displays benchmark governance disclosure on model reality and data origins."""
    html = """
    <div class="alert-box alert-info" style="font-size: 0.86rem;">
        <strong>🔬 BENCHMARK GOVERNANCE NOTICE — SYNTHETIC DATASET DERIVATION:</strong><br>
        The headline evaluation scores (Macro F1 = 0.697, R² = 0.970) reflect reverse-engineering of the dataset generator's
        deterministic piecewise formulas. Real-world lending involves compounding interest rates, non-linear bureau delinquency gradients,
        and macroeconomic employment shocks not captured in static synthetic data. These models serve as an intelligent decision-support
        demonstrator, not an autonomous banking underwriting engine.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
