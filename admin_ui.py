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

# --- FIXED CSS FOR READABILITY ---
st.markdown("""
    <style>
    /* 1. Force Dark Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* 2. Force All Text to be Light/White */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #E0E0E0 !important;
    }

    /* 3. Exceptions: Metric Cards & Code Blocks need specific colors */
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4F46E5;
        margin-bottom: 10px;
    }
    /* Fix text inside Metric Cards to be readable */
    .metric-card b, .metric-card span {
        color: #E0E0E0 !important;
    }
    /* Green text for latency */
    .latency-text {
        color: #4ade80 !important;
    }

    /* 4. Chat Message Styling */
    /* User Message Bubble */
    div[data-testid="stChatMessage"] {
        background-color: #262730;
        border: 1px solid #41444e;
        border-radius: 10px;
    }
    /* User Avatar */
    div[data-testid="stChatMessage"] svg {
        fill: #E0E0E0 !important;
    }

    /* 5. Warning/Info Boxes (Restore readability) */
    .stAlert {
        color: white !important;
    }
    
    /* 6. Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🟢 System Status")
    st.info("Connected to Vertex AI & BigQuery")
    
    st.divider()
    
    st.markdown("### 🔍 Observability")
    st.markdown("Traces are being sent to **Datadog**.")
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
    
    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a tax question..."):
        # 1. Show User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call API
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
                <b>⏱️ Latency</b><br>
                <span class="latency-text" style="font-size: 24px;">{round(meta['latency'], 2)} ms</span>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <b>🤖 Model</b><br>
                <span style="font-size: 18px;">{meta['model']}</span>
            </div>
            """, unsafe_allow_html=True)

        # 2. Tabs
        tab_rag, tab_bq, tab_json = st.tabs(["📚 RAG Sources", "📈 BigQuery Data", "⚙️ JSON"])
        
        with tab_rag:
            if meta['sources'] and len(meta['sources']) > 0:
                for i, src in enumerate(meta['sources']):
                    with st.expander(f"📄 Document Chunk #{i+1}", expanded=(i==0)):
                        st.text(src) # Use st.text to avoid markdown parsing errors
            else:
                st.warning("No specific documents found. LLM used general knowledge.")

        with tab_bq:
            if meta['economic_data']:
                st.info(meta['economic_data'])
            else:
                st.text("No economic context requested for this query.")

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