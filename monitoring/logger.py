"""
Monitoring logger — logs every query to a SQLite database.
Tracks: timestamp, question, answer, latency, number of sources returned.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries.db")


def init_db():
    """Create the queries table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            latency_ms  REAL NOT NULL,
            num_sources INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_query(question: str, answer: str, latency_ms: float, num_sources: int):
    """Log a single query to the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO queries (timestamp, question, answer, latency_ms, num_sources) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), question, answer, latency_ms, num_sources),
    )
    conn.commit()
    conn.close()


def get_all_queries() -> list[dict]:
    """Fetch all logged queries as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM queries ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats() -> dict:
    """Return summary statistics."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT
            COUNT(*)            AS total_queries,
            ROUND(AVG(latency_ms), 2)  AS avg_latency_ms,
            ROUND(MIN(latency_ms), 2)  AS min_latency_ms,
            ROUND(MAX(latency_ms), 2)  AS max_latency_ms,
            ROUND(AVG(num_sources), 2) AS avg_sources
        FROM queries
    """).fetchone()
    conn.close()
    return {
    "total_queries": row[0],
    "avg_latency_ms": row[1],
    "min_latency_ms": row[2],
    "max_latency_ms": row[3],
    "avg_sources": row[4],
} if row else {}


# Initialise DB on import
init_db()