"""
Prediction History View: Comprehensive Audit Trail & SQLite CRUD Interface.
"""

import json
import streamlit as st
import pandas as pd

from app.db import (
    get_predictions,
    get_prediction_by_id,
    update_prediction_notes,
    delete_prediction,
    clear_all_predictions,
    get_history_summary_stats
)
from app.components import status_badge_html


def render_history_view():
    """Renders the prediction audit history interface with full CRUD capabilities."""
    st.markdown('<div class="brand-title" style="font-size: 2rem; font-weight: 700;">'
                'Underwriting Audit <span class="brand-gradient">History & Telemetry</span>'
                '</div>', unsafe_allow_html=True)
    st.caption("Immutable local ledger of underwriting decisions and credit capacity estimations.")

    stats = get_history_summary_stats()

    # Filter & Search Controls
    f_c1, f_c2, f_c3 = st.columns([1.5, 1.5, 2.0])
    with f_c1:
        type_filter = st.selectbox("Filter by Assessment Type:", ["All", "Eligibility", "Capacity", "Joint"])
    with f_c2:
        class_filter = st.selectbox("Filter by Eligibility Result:", ["All", "Eligible", "High_Risk", "Not_Eligible"])
    with f_c3:
        record_limit = st.slider("Display Limit (Recent records):", min_value=10, max_value=250, value=50, step=10)

    search_query = st.text_input("🔍 Search Audit Records (by ID, Loan Purpose, or Notes):", placeholder="e.g. Personal Loan, High_Risk, #12")

    # Fetch records
    df_records = get_predictions(
        limit=record_limit,
        assessment_type=type_filter if type_filter != "All" else None,
        eligibility_filter=class_filter if class_filter != "All" else None,
        search_query=search_query
    )

    if df_records.empty:
        st.info("ℹ️ No historical records found matching the active filters or search query. Run an assessment from the sidebar to populate history.")
        return

    st.markdown(f"**Showing {len(df_records)} historical assessment records (of {stats['total_records']} total logged):**")

    # Display clean table
    display_cols = [
        "id", "timestamp", "assessment_type", "monthly_salary",
        "requested_amount", "requested_tenure", "predicted_eligibility",
        "predicted_max_emi_policy", "foir_percentage"
    ]
    present_cols = [c for c in display_cols if c in df_records.columns]

    st.dataframe(
        df_records[present_cols].rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "assessment_type": "Type",
            "monthly_salary": "Salary (₹)",
            "requested_amount": "Loan (₹)",
            "requested_tenure": "Tenure (m)",
            "predicted_eligibility": "Decision",
            "predicted_max_emi_policy": "Max EMI (₹)",
            "foir_percentage": "FOIR (%)"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Action Toolbar: Export, Inspect & Delete
    st.markdown("---")
    st.markdown("### 🛠️ Record Inspection & Management")

    act_c1, act_c2 = st.columns([1.5, 1.5])

    # 1. Inspect Individual Record & Update Notes (CRUD: Read & Update)
    with act_c1:
        st.markdown("#### 🔍 Inspect & Update Record")
        record_ids = df_records["id"].tolist()
        selected_id = st.selectbox("Select Record ID to Inspect:", record_ids)

        if selected_id:
            record_dict = get_prediction_by_id(selected_id)
            if record_dict:
                st.write(f"**Record ID:** `#{record_dict['id']}` | **Logged:** `{record_dict['timestamp']}`")
                st.write(f"• **Type:** `{record_dict['assessment_type']}` | **Decision:** {status_badge_html(record_dict['predicted_eligibility'] or 'N/A')}", unsafe_allow_html=True)
                st.write(f"• **Salary:** ₹{record_dict['monthly_salary']:,.2f} | **Loan:** ₹{record_dict['requested_amount']:,.2f} over {record_dict['requested_tenure']}m")
                st.write(f"• **Policy Max EMI:** ₹{record_dict['predicted_max_emi_policy']:,.2f} (Raw: ₹{record_dict['predicted_max_emi_raw']:,.2f})")
                st.write(f"• **Projected FOIR:** {record_dict['foir_percentage']:.1f}% | **Implied Installment (P/N, 0% APR):** ₹{record_dict['implied_installment']:,.2f}")
                if record_dict.get("eligibility_probabilities"):
                    try:
                        probs = json.loads(record_dict["eligibility_probabilities"])
                        st.write("**Class Probabilities:**", probs)
                    except Exception:
                        pass

                # Underwriter remarks / Notes update form
                current_notes = record_dict.get("notes") or ""
                new_notes = st.text_input("Underwriter Remarks / Audit Notes:", value=current_notes, key=f"notes_{selected_id}")
                if st.button("💾 Save Underwriter Notes", use_container_width=True):
                    upd_ok = update_prediction_notes(selected_id, new_notes)
                    if upd_ok:
                        st.success(f"Audit notes for record #{selected_id} updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update notes.")

    # 2. Deletion & Export Actions (CRUD: Delete)
    with act_c2:
        st.markdown("#### 🗑️ Audit Ledger Operations")

        if st.button("Delete Selected Record", type="secondary", use_container_width=True):
            if selected_id:
                success = delete_prediction(selected_id)
                if success:
                    st.success(f"Deleted record #{selected_id}!")
                    st.rerun()
                else:
                    st.error("Failed to delete record.")

        # Clear All History with Confirmation
        st.markdown("##### Dangerous Operations")
        confirm_clear = st.checkbox("Confirm: Permanently erase ALL prediction history records")
        if st.button("Clear All History", type="primary", disabled=not confirm_clear, use_container_width=True):
            count = clear_all_predictions()
            st.warning(f"Cleared {count} records from the database.")
            st.rerun()

        # Export Options
        st.markdown("##### Data Export")
        csv_data = df_records.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Ledger to CSV",
            data=csv_data,
            file_name=f"emipredict_audit_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
