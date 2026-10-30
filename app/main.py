"""
EMIPredict AI — Streamlit Web Application Main Entry Point.
Intelligent Multi-Task Financial Risk Assessment & EMI Capacity Underwriting Platform.
"""

import sys
import logging
from pathlib import Path
import joblib
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="EMIPredict AI — Financial Risk Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path if not present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.components import inject_custom_css
from app.db import init_db
from app.views.dashboard import render_dashboard_view
from app.views.eligibility import render_eligibility_view
from app.views.capacity import render_capacity_view
from app.views.joint import render_joint_view
from app.views.history import render_history_view
from app.views.model_performance import render_model_performance_view

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource
def load_production_models():
    """
    Loads serialized production pipelines and label encoder.
    Cached across sessions for maximum responsiveness.
    """
    clf_path = Path("models/classification/best_application_model.joblib")
    reg_path = Path("models/regression/best_application_model.joblib")
    le_path = Path("models/classification/label_encoder.joblib")

    if not clf_path.exists():
        raise FileNotFoundError(f"Missing classification model artifact: {clf_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing regression model artifact: {reg_path}")
    if not le_path.exists():
        raise FileNotFoundError(f"Missing label encoder artifact: {le_path}")

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    le = joblib.load(le_path)
    logger.info("Loaded production model artifacts successfully.")
    return clf, reg, le


def main():
    """Main application lifecycle and page routing."""
    inject_custom_css()
    init_db()

    # Load pipelines
    try:
        clf_pipe, reg_pipe, le = load_production_models()
    except Exception as e:
        st.error(f"❌ Critical Error loading production model artifacts: {e}")
        st.stop()

    # Sidebar Navigation
    st.sidebar.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; color: #F8FAFC;">
                💳 <span style="background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">EMIPredict AI</span>
            </div>
            <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 2px;">Intelligent Underwriting Engine</div>
        </div>
    """, unsafe_allow_html=True)

    nav_selection = st.sidebar.radio(
        "Navigation Menu:",
        [
            "📊 Dashboard",
            "🎯 EMI Eligibility Prediction",
            "📐 Maximum EMI Capacity",
            "⚡ Unified Risk & Capacity",
            "📜 Underwriting History",
            "🛡️ Model Governance & About"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="font-size: 0.78rem; color: #64748B; line-height: 1.5;">
            <strong>System Telemetry:</strong><br>
            • Ensembles: XGBoost Balanced<br>
            • Test Split: 80,960 records<br>
            • Gen Gap: &lt; 0.36%<br>
            • Storage: SQLite WAL Ledger<br>
            • Phase: Phase 12 Verified Production UI
        </div>
    """, unsafe_allow_html=True)

    # Page Router
    if nav_selection == "📊 Dashboard":
        render_dashboard_view()
    elif nav_selection == "🎯 EMI Eligibility Prediction":
        render_eligibility_view(clf_pipe, le)
    elif nav_selection == "📐 Maximum EMI Capacity":
        render_capacity_view(reg_pipe)
    elif nav_selection == "⚡ Unified Risk & Capacity":
        render_joint_view(clf_pipe, reg_pipe, le)
    elif nav_selection == "📜 Underwriting History":
        render_history_view()
    elif nav_selection == "🛡️ Model Governance & About":
        render_model_performance_view()


if __name__ == "__main__":
    main()
