import streamlit as st
import requests
import json
import time
import os

# CONFIGURATION
# If running inside Docker Compose, use the service name.
# If running locally while API is in Docker, use localhost.
API_URL = os.getenv("API_URL", "https://betawork.onrender.com/ask") 

st.set_page_config(
    page_title="BetaBot Logic Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS HACKS FOR "HACKATHON VIBE" ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4F46E5;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: DATADOG STATUS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/25/Datadog_logo.svg", width=150)
    st.markdown("### 🟢 System Status")
    
    st.metric(label="Service", value="betawork-ai-engine")
    st.metric(label="RAG Vector DB", value="Vertex AI Search")
    st.metric(label="Economic Data", value="BigQuery Public")
    
    st.divider()
    
    st.markdown("### 🔍 Observability")
    st.info("Traces are being sent to Datadog Agent.")
    st.link_button("View Datadog Dashboard", "https://app.datadoghq.com/apm/traces")

# --- MAIN LAYOUT ---
st.title("🧠 BetaBot: Regulatory Command Center")
st.markdown("*Hackathon Admin View: Analyzing reasoning, retrieval, and latency.*")

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
    if prompt := st.chat_input("Ask a tax question (e.g. Is lunch allowance taxable?)"):
        # 1. Show User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call API
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("▌ *Thinking (Querying Vector DB)...*")
            
            try:
                # Prepare Payload (Tax Mode)
                payload = {"query": prompt, "mode": "tax"}
                
                # Request
                start_ts = time.time()
                response = requests.post(API_URL, json=payload)
                end_ts = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    
                    # Store Metadata for Right Column
                    st.session_state.last_metadata = {
                        "sources": data.get("sources", []),
                        "economic_data": data.get("economic_data", ""),
                        "latency": data.get("latency_ms", (end_ts - start_ts)*1000),
                        "model": data.get("model_used", "gemini-2.5-pro")
                    }
                    
                    # Typewriter Effect
                    full_response = ""
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Force refresh to update Right Column
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
                <span style="font-size: 24px; color: #4ade80">{round(meta['latency'], 2)} ms</span>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <b>🤖 Model</b><br>
                <span style="font-size: 18px;">{meta['model']}</span>
            </div>
            """, unsafe_allow_html=True)

        # 2. Tabs for Evidence
        tab_rag, tab_bq, tab_json = st.tabs(["📚 RAG Sources", "📈 BigQuery Data", "⚙️ Raw JSON"])
        
        with tab_rag:
            st.caption("Legal Documents retrieved from Vertex AI Search")
            if meta['sources']:
                for i, src in enumerate(meta['sources']):
                    with st.expander(f"📄 Document Chunk #{i+1}", expanded=(i==0)):
                        st.code(src, language="text")
            else:
                st.warning("No specific documents found. LLM used general knowledge.")

        with tab_bq:
            st.caption("Economic Indicators from BigQuery Public Datasets")
            if meta['economic_data']:
                st.info(meta['economic_data'])
                st.bar_chart({"Inflation": 24.5, "GDP Growth": 3.1}) # Mock chart viz of the data
            else:
                st.text("No economic context requested for this query.")

        with tab_json:
            st.json(meta)
            
    else:
        st.info("Waiting for query... Ask a question on the left to see the reasoning engine.")
        st.markdown("""
        **What happens when you ask?**
        1. **FastAPI** receives query + Datadog Trace ID.
        2. **Vertex AI Search** retrieves top 3 tax PDF chunks.
        3. **BigQuery** fetches live Nigeria GDP/Inflation data.
        4. **Gemini 2.5 Pro** synthesizes answer.
        5. **Streamlit** visualizes the evidence.
        """)