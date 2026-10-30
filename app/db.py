"""
SQLite Persistence Layer for Prediction History in EMIPredict AI.

Features:
- Safe, parameterized queries to prevent SQL injection.
- Privacy-compliant: stores only non-PII operational financial inputs and model predictions.
- Full CRUD operations: Insert, Read/Filter, Delete, Clear, and Summary Aggregations.
- Export-ready DataFrame transformations.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/predictions_history.db")


def get_db_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initializes the prediction_history table schema if it does not exist."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                assessment_type TEXT NOT NULL,
                age REAL,
                monthly_salary REAL,
                requested_amount REAL,
                requested_tenure REAL,
                emi_scenario TEXT,
                credit_score REAL,
                foir_percentage REAL,
                implied_installment REAL,
                predicted_eligibility TEXT,
                eligibility_probabilities TEXT,
                predicted_max_emi_raw REAL,
                predicted_max_emi_policy REAL,
                capacity_margin REAL,
                notes TEXT
            )
        """)
        # Index on timestamp and assessment_type for fast filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_history_timestamp
            ON prediction_history (timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_history_type
            ON prediction_history (assessment_type)
        """)
        conn.commit()
    logger.info("Database initialized at %s", db_path)


def save_prediction(
    assessment_type: str,
    applicant_data: Dict[str, Any],
    ratios: Dict[str, Any],
    eligibility_result: Optional[str] = None,
    probabilities: Optional[Dict[str, float]] = None,
    max_emi_raw: Optional[float] = None,
    max_emi_policy: Optional[float] = None,
    capacity_margin: Optional[float] = None,
    notes: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH
) -> int:
    """
    Saves a prediction record into the SQLite database.
    Does NOT store sensitive applicant identifiers (no names, phones, PAN, accounts).
    """
    init_db(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    probs_json = json.dumps(probabilities) if probabilities else None

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history (
                timestamp,
                assessment_type,
                age,
                monthly_salary,
                requested_amount,
                requested_tenure,
                emi_scenario,
                credit_score,
                foir_percentage,
                implied_installment,
                predicted_eligibility,
                eligibility_probabilities,
                predicted_max_emi_raw,
                predicted_max_emi_policy,
                capacity_margin,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str,
            assessment_type,
            float(applicant_data.get("age", 0.0)),
            float(applicant_data.get("monthly_salary", 0.0)),
            float(applicant_data.get("requested_amount", 0.0)),
            float(applicant_data.get("requested_tenure", 0.0)),
            str(applicant_data.get("emi_scenario", "")),
            float(applicant_data.get("credit_score", 0.0)),
            float(ratios.get("projected_foir", 0.0)),
            float(ratios.get("implied_installment", 0.0)),
            eligibility_result,
            probs_json,
            float(max_emi_raw) if max_emi_raw is not None else None,
            float(max_emi_policy) if max_emi_policy is not None else None,
            float(capacity_margin) if capacity_margin is not None else None,
            notes
        ))
        conn.commit()
        record_id = cursor.lastrowid
    return int(record_id)


def get_predictions(
    db_path: Path = DEFAULT_DB_PATH,
    limit: int = 100,
    assessment_type: Optional[str] = None,
    eligibility_filter: Optional[str] = None,
    search_query: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetches prediction records with optional filtering and keyword search.
    Returns a pandas DataFrame formatted for table display.
    """
    init_db(db_path)
    query = "SELECT * FROM prediction_history WHERE 1=1"
    params: List[Any] = []

    if assessment_type and assessment_type.lower() != "all":
        query += " AND assessment_type = ?"
        params.append(assessment_type)

    if eligibility_filter and eligibility_filter.lower() != "all":
        query += " AND predicted_eligibility = ?"
        params.append(eligibility_filter)

    if search_query and search_query.strip():
        s = f"%{search_query.strip()}%"
        query += " AND (notes LIKE ? OR emi_scenario LIKE ? OR CAST(id AS TEXT) LIKE ?)"
        params.extend([s, s, s])

    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    try:
        with get_db_connection(db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df
    except sqlite3.Error as e:
        logger.error("Error executing get_predictions query: %s", e)
        return pd.DataFrame()


def get_prediction_by_id(pred_id: Any, db_path: Path = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a single prediction by primary key. Safely casts pred_id to int."""
    init_db(db_path)
    try:
        clean_id = int(pred_id)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prediction_history WHERE id = ?", (clean_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
    except (ValueError, TypeError, sqlite3.Error) as e:
        logger.error("Error retrieving prediction by ID %s: %s", pred_id, e)
    return None


def update_prediction_notes(pred_id: Any, notes: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """
    Updates the underwriting notes/remarks for a specific prediction record.
    Supports full CRUD lifecycle requirements.
    """
    init_db(db_path)
    try:
        clean_id = int(pred_id)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE prediction_history SET notes = ? WHERE id = ?", (str(notes), clean_id))
            conn.commit()
            return cursor.rowcount > 0
    except (ValueError, TypeError, sqlite3.Error) as e:
        logger.error("Error updating prediction notes for ID %s: %s", pred_id, e)
        return False


def delete_prediction(pred_id: Any, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Deletes a single prediction record by ID. Safely casts pred_id to int."""
    init_db(db_path)
    try:
        clean_id = int(pred_id)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_history WHERE id = ?", (clean_id,))
            conn.commit()
            return cursor.rowcount > 0
    except (ValueError, TypeError, sqlite3.Error) as e:
        logger.error("Error deleting prediction ID %s: %s", pred_id, e)
        return False


def clear_all_predictions(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Clears all records from the prediction_history table."""
    init_db(db_path)
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_history")
            deleted_count = cursor.rowcount
            conn.commit()
        return int(deleted_count)
    except sqlite3.Error as e:
        logger.error("Error clearing prediction history: %s", e)
        return 0


def get_history_summary_stats(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Computes summary statistics over all historical predictions."""
    init_db(db_path)
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prediction_history")
            total_count = cursor.fetchone()[0]

            cursor.execute("""
                SELECT predicted_eligibility, COUNT(*)
                FROM prediction_history
                WHERE predicted_eligibility IS NOT NULL
                GROUP BY predicted_eligibility
            """)
            eligibility_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("""
                SELECT AVG(predicted_max_emi_policy), AVG(foir_percentage), AVG(requested_amount)
                FROM prediction_history
            """)
            avg_row = cursor.fetchone()
            avg_emi = avg_row[0] if avg_row[0] is not None else 0.0
            avg_foir = avg_row[1] if avg_row[1] is not None else 0.0
            avg_amount = avg_row[2] if avg_row[2] is not None else 0.0

        return {
            "total_records": total_count,
            "eligibility_distribution": eligibility_counts,
            "average_max_emi": round(float(avg_emi), 2),
            "average_foir": round(float(avg_foir), 2),
            "average_requested_amount": round(float(avg_amount), 2)
        }
    except sqlite3.Error as e:
        logger.error("Error computing history summary stats: %s", e)
        return {
            "total_records": 0,
            "eligibility_distribution": {},
            "average_max_emi": 0.0,
            "average_foir": 0.0,
            "average_requested_amount": 0.0
        }
