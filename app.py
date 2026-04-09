import streamlit as st
from rag_engine import initialize_rag
 
st.set_page_config(
    page_title="Placement RAG Assistant",
    page_icon="🎓",
    layout="centered",
)
 
st.title("🎓 Placement Assistant")
st.caption("Ask me anything about company placements — roles, CTC, CGPA cutoffs, internships & more.")
 
 
@st.cache_resource(show_spinner="Setting up the placement database...")
def load_rag():
    return initialize_rag()
 
 
# ── Load RAG (show a friendly error if setup fails) ─────────────────────────
try:
    rag_chain, retriever = load_rag()
except EnvironmentError as e:
    st.error(f"⚙️ Configuration error: {e}")
    st.stop()
except FileNotFoundError as e:
    st.error(f"📁 Dataset not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error during startup: {e}")
    st.stop()
 
# ── Chat history ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your Placement Assistant 👋\n\n"
                "I have data on **200 companies** — CTC, roles, CGPA cutoffs, "
                "internship availability, and sectors.\n\n"
                "Try asking:\n"
                "- *Which companies offer internships with CTC above 20 LPA?*\n"
                "- *What is the CGPA cutoff for Google?*\n"
                "- *Show me fintech companies with minimum 7.0 CGPA*"
            ),
        }
    ]
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# ── Chat input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about placements..."):
 
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        with st.spinner("Searching placement database..."):
            # BUG FIX: rag_chain now returns a dict with 'answer' and
            # 'source_docs' in one pass — no double vector-store hit.
            result = rag_chain.invoke(user_input)
            answer = result["answer"]
            source_docs = result["source_docs"]
 
        st.markdown(answer)
 
        if source_docs:
            with st.expander("📂 Sources retrieved from database"):
                seen = set()
                for doc in source_docs:
                    company    = doc.metadata.get("company", "")
                    role       = doc.metadata.get("role", "")
                    ctc        = doc.metadata.get("ctc", "")
                    cgpa       = doc.metadata.get("cgpa", "")
                    internship = doc.metadata.get("internship", "")
                    sector     = doc.metadata.get("sector", "")
                    key = f"{company}-{role}"
                    if key not in seen:
                        seen.add(key)
                        st.markdown(
                            f"**{company}** — {role} | "
                            f"CTC: ₹{ctc} LPA | "
                            f"Min CGPA: {cgpa} | "
                            f"Internship: {internship} | "
                            f"Sector: {sector}"
                        )
 
    st.session_state.messages.append({"role": "assistant", "content": answer})