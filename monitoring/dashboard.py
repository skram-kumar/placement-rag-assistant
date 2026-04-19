"""
Monitoring Dashboard — visualizes query logs from SQLite.
Run with: streamlit run monitoring/dashboard.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from monitoring.logger import get_all_queries, get_stats

st.set_page_config(
    page_title="RAG Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Placement RAG — Monitoring Dashboard")
st.caption("Live query logs and performance metrics from the FastAPI backend.")

# ── Auto refresh ──────────────────────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.rerun()

# ── Stats ─────────────────────────────────────────────────────────────────────
stats = get_stats()

if not stats or stats.get("total_queries") == 0:
    st.info("No queries logged yet. Send some queries via the API at http://localhost:8000/docs")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Queries", stats["total_queries"])
col2.metric("Avg Latency", f"{stats['avg_latency_ms']} ms")
col3.metric("Min Latency", f"{stats['min_latency_ms']} ms")
col4.metric("Max Latency", f"{stats['max_latency_ms']} ms")

st.divider()

# ── Query logs ────────────────────────────────────────────────────────────────
queries = get_all_queries()
df = pd.DataFrame(queries)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.rename(columns={
    "timestamp": "Time",
    "question": "Question",
    "answer": "Answer",
    "latency_ms": "Latency (ms)",
    "num_sources": "Sources",
})

# ── Latency chart ─────────────────────────────────────────────────────────────
st.subheader("⚡ Latency Over Time")
st.line_chart(df.set_index("Time")["Latency (ms)"])

st.divider()

# ── Query table ───────────────────────────────────────────────────────────────
st.subheader("🗂 Query Log")
st.dataframe(
    df[["Time", "Question", "Latency (ms)", "Sources"]],
    use_container_width=True,
    hide_index=True,
)

# ── Expandable answers ────────────────────────────────────────────────────────
st.divider()
st.subheader("💬 Full Answers")
for _, row in df.iterrows():
    with st.expander(f"{row['Time'].strftime('%H:%M:%S')} — {row['Question'][:80]}"):
        st.markdown(f"**Latency:** {row['Latency (ms)']} ms | **Sources:** {row['Sources']}")
        st.markdown(row["Answer"])