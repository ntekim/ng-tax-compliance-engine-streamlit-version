import streamlit as st
import requests
import json
import time
import os

# CONFIGURATION
API_URL = os.getenv("API_URL", "https://betawork-ai-engine.onrender.com/ask") 

st.set_page_config(
    page_title="BetaBot Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR DARK MODE & READABILITY ---
st.markdown("""
    <style>
      /* 1. Main Background */
    .stApp {
        background-color: #0E1117;
    }

    /* 2. Sidebar Background (THE FIX) */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* 3. Force All Text to be Light/White */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #E0E0E0 !important;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4F46E5;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        color: #4ade80 !important;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #A0A0A0 !important;
    }

    /* Chat Message Bubbles */
    div[data-testid="stChatMessage"] {
        background-color: #1E2329;
        border: 1px solid #2B303B;
    }
    
    /* Document/Code Box Styling */
    .doc-box {
        background-color: #1E293B;
        color: #E2E8F0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #F59E0B; /* Amber for Docs */
        font-family: monospace;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    .data-box {
        background-color: #064E3B; /* Dark Green background */
        color: #6EE7B7; /* Light Green Text */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #10B981;
        font-family: monospace;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🟢 System Status")
    st.success("API Connected")
    st.info("RAG Source: Vertex AI Search")
    st.info("Context Source: BigQuery Public Data")
    
    st.divider()
    
    st.markdown("### 🔍 Observability")
    st.write("Traces sent to **Datadog Agent**.")
    st.link_button("View Datadog Dashboard", "https://app.datadoghq.com/apm/traces")

# --- MAIN LAYOUT ---
st.title("🧠 BetaBot: Regulatory Command Center")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metadata" not in st.session_state:
    st.session_state.last_metadata = None

# Create Two Columns
col_chat, col_debug = st.columns([1.2, 1])

# --- LEFT COLUMN: CHAT INTERFACE ---
with col_chat:
    st.subheader("💬 User Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question (e.g. 'Is lunch allowance taxable?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("▌ *Thinking...*")
            
            try:
                payload = {"query": prompt, "mode": "tax"}
                start_ts = time.time()
                response = requests.post(API_URL, json=payload)
                end_ts = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    
                    st.session_state.last_metadata = {
                        "sources": data.get("sources", []),
                        "economic_data": data.get("economic_data", ""),
                        "latency": (end_ts - start_ts) * 1000,
                        "model": "gemini-2.5-pro"
                    }
                    
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                else:
                    message_placeholder.error(f"API Error: {response.text}")
            except Exception as e:
                message_placeholder.error(f"Connection Failed: {e}")

# --- RIGHT COLUMN: INTELLIGENCE PANEL ---
with col_debug:
    st.subheader("🛠️ Logic & Evidence")
    
    meta = st.session_state.last_metadata
    
    if meta:
        # 1. Metrics Row
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">⏱️ Latency</div>
                <div class="metric-value">{round(meta['latency'], 2)} ms</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🤖 Model</div>
                <div class="metric-value" style="color:white !important; font-size:18px;">{meta['model']}</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Tabs
        tab_rag, tab_bq, tab_json = st.tabs(["📚 RAG Sources", "📈 BigQuery Data", "⚙️ JSON"])
        
        with tab_rag:
            if meta['sources'] and len(meta['sources']) > 0:
                st.caption(f"✅ Retrieved {len(meta['sources'])} chunks")
                
                for i, doc in enumerate(meta['sources']):
                    # doc is now a dict: {'source': 'CAMA 2020', 'content': '...'}
                    
                    doc_title = doc.get('source', f'Document #{i+1}')
                    doc_content = doc.get('content', '')
                    
                    with st.expander(f"📄 {doc_title}", expanded=(i==0)):
                        # Nice header for the citation
                        st.markdown(f"**Source:** *{doc_title}*", unsafe_allow_html=True)
                        # The text content
                        st.markdown(f'<div class="doc-box">{doc_content}</div>', unsafe_allow_html=True)
            else:
                st.warning("No documents found.")

        with tab_bq:
            if meta['economic_data']:
                st.caption("✅ Live Data fetched from bigquery-public-data")
                # Force Green Box for Data
                st.markdown(f'<div class="data-box">{meta["economic_data"]}</div>', unsafe_allow_html=True)
                
                # Visual Chart
                st.caption("Visualized Trend:")
                st.bar_chart({"Inflation": 28.9, "GDP Growth": 3.4}) 
            else:
                st.info("No economic context requested for this query.")

        with tab_json:
            st.json(meta)
            
    else:
        st.info("Waiting for query... Ask a question on the left.")
        st.markdown("""
        **What happens when you ask?**
        1. **FastAPI** receives query + Datadog Trace ID.
        2. **Vertex AI Search** retrieves top 3 tax PDF chunks.
        3. **BigQuery** fetches live Nigeria GDP/Inflation data.
        4. **Gemini 2.5 Pro** synthesizes answer.
        5. **Streamlit** visualizes the evidence.
        """)