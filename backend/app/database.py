"""
Database module for SQLite logging of predictions.
"""
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Database path
DB_DIR = Path(__file__).parent.parent
DB_PATH = DB_DIR / "predictions.db"


def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_used TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            input_summary TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def log_prediction(model_used: str, predicted_price: float, input_data: Dict[str, Any]):
    """
    Log a prediction to the database.
    
    Args:
        model_used: Model name used for prediction
        predicted_price: Predicted price
        input_data: Input features (sanitized, no PII)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.utcnow().isoformat()
    input_summary = json.dumps(input_data)
    
    cursor.execute("""
        INSERT INTO predictions (timestamp, model_used, predicted_price, input_summary)
        VALUES (?, ?, ?, ?)
    """, (timestamp, model_used, predicted_price, input_summary))
    
    conn.commit()
    conn.close()
    logger.info(f"Logged prediction: ${predicted_price:,.2f} using {model_used}")


def get_recent_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent prediction logs.
    
    Args:
        limit: Maximum number of logs to return
    
    Returns:
        List of log entries
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, timestamp, model_used, predicted_price, input_summary
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for row in rows:
        logs.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "model_used": row["model_used"],
            "predicted_price": row["predicted_price"],
            "input_summary": json.loads(row["input_summary"])
        })
    
    return logs
