import os
import json
import base64
import logging
import uvicorn
import vertexai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions

# --- 1. CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betabot")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "silver-impulse-481722-v5")
# The App ID from your screenshot (Ending in 13359) used for path construction
ENGINE_ID = "nigeria-compliance-engine_1766620713359" 
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
CREDENTIALS_FILE = os.path.abspath("gcp_key.json")

# --- 2. AUTH SETUP ---
my_credentials = None
if os.getenv("GCP_CREDENTIALS_BASE64"):
    try:
        decoded_key = base64.b64decode(os.getenv("GCP_CREDENTIALS_BASE64"))
        with open(CREDENTIALS_FILE, "w") as f:
            f.write(decoded_key.decode("utf-8"))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
        my_credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    except Exception as e:
        logger.error(f"Auth Error: {e}")

# --- 3. INIT CLIENTS ---
model = None
bq_client = None
try:
    # Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro")
    
    # BigQuery
    bq_client = bigquery.Client(project=PROJECT_ID, credentials=my_credentials)
except Exception as e:
    logger.error(f"Service Init Error: {e}")

# --- 4. DATADOG DUMMY ---
try:
    from ddtrace import tracer, patch_all
    from ddtrace.contrib.fastapi import TraceMiddleware
    patch_all()
    DD_ENABLED = True
except:
    DD_ENABLED = False
    class DummyTracer:
        def wrap(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
    tracer = DummyTracer()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if DD_ENABLED:
    app.add_middleware(TraceMiddleware, service="betawork-ai-engine")

# --- 5. DATA MODELS ---
class QueryRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    mode: str = "tax"
    @model_validator(mode='before')
    @classmethod
    def check_query_or_question(cls, data):
        if not data.get('query') and data.get('question'):
            data['query'] = data['question']
        return data

class APIResponse(BaseModel):
    answer: str
    sources: List[str]

# --- 6. LOGIC ---

def get_economic_context():
    """Fetches Live GDP/Inflation data from Public BigQuery"""
    if not bq_client: return ""
    try:
        query = """
            SELECT indicator_name, value, year
            FROM `bigquery-public-data.world_bank_wdi.indicators`
            WHERE country_code = 'NGA'
            AND indicator_code IN ('NY.GDP.MKTP.KD.ZG', 'FP.CPI.TOTL.ZG')
            ORDER BY year DESC LIMIT 3
        """
        results = bq_client.query(query).result()
        stats = [f"- {row.indicator_name} ({row.year}): {row.value:.2f}%" for row in results]
        return "\n".join(stats)
    except Exception as e:
        logger.error(f"BigQuery Error: {e}")
        return ""

@tracer.wrap(name="rag_search", service="betawork-ai-engine")
def search_documents(query: str):
    if not my_credentials: return []

    try:
        client_opts = ClientOptions(api_endpoint="discoveryengine.googleapis.com")
        client = discoveryengine.SearchServiceClient(credentials=my_credentials, client_options=client_opts)
        
        # Use the Engine path which wraps the Data Store
        serving_config = (
            f"projects/{PROJECT_ID}/locations/global/collections/default_collection/"
            f"engines/{ENGINE_ID}/servingConfigs/default_search"
        )
        
        req = discoveryengine.SearchRequest(
            serving_config=serving_config, 
            query=query, 
            page_size=3,
            # Enable Spell Check & Query Expansion to find PDF text better
            query_expansion_spec={"condition": "AUTO"},
            spell_correction_spec={"mode": "AUTO"}
        )
        
        response = client.search(req)
        
        sources = []
        for result in response.results:
            data = result.document.derived_struct_data
            
            # 1. Try Extractive Segments (Best for PDFs)
            if 'extractive_segments' in data:
                for seg in data['extractive_segments']:
                    sources.append(seg.get('content', ''))
            
            # 2. Try Snippets (Fallback)
            elif 'snippets' in data:
                for snip in data['snippets']:
                    sources.append(snip.get('snippet', ''))
            
        return sources[:3] # Return top 3 chunks
    except Exception as e:
        logger.error(f"Search API Error: {e}")
        return []

@tracer.wrap(name="generate_answer", service="betawork-ai-engine")
def get_ai_response(user_query: str, mode: str):
    if not model: return "AI System Offline.", []
    
    # Mode 1: Therapy
    if mode == "therapy":
        prompt = f"You are BetaCare, a workplace therapist. User: '{user_query}'. Be empathetic, warm, and professional."
        try:
            return model.generate_content(prompt).text, []
        except:
            return "I am listening.", []

    # Mode 2: Business/Tax
    
    # A. Fetch Context
    sources = search_documents(user_query)
    econ_data = get_economic_context()
    
    # B. Build Context String
    rag_text = ""
    if sources:
        rag_text = "OFFICIAL DOCUMENTS FOUND:\n" + "\n---\n".join(sources)
    
    # C. The "Balanced" Prompt
    prompt = f"""
    You are BetaBot, an intelligent Business & Tax Advisor for Nigeria.
    
    ECONOMIC CONTEXT:
    {econ_data}
    
    {rag_text}
    
    USER QUESTION: "{user_query}"
    
    INSTRUCTIONS:
    1. Answer the user's question clearly and professionally.
    2. If the user asks about a general business concept (e.g., "How to market?", "What is cash flow?"), answer broadly but mention how it applies in Nigeria if relevant.
    3. If the user asks about TAX or LAW, prioritize the 'OFFICIAL DOCUMENTS' provided above.
    4. If no documents are found, use your general knowledge of FIRS, CAMA 2020, and Nigerian business practices.
    """

    try:
        response = model.generate_content(prompt)
        return response.text, sources
    except Exception as e:
        return f"Thinking Error: {e}", []

# --- ENDPOINTS ---
@app.get("/")
def root(): return {"status": "running"}

@app.post("/ask", response_model=APIResponse)
async def ask_endpoint(req: QueryRequest):
    final_query = req.query or req.question
    if not final_query: raise HTTPException(status_code=400, detail="Query required")
    
    answer, sources = get_ai_response(final_query, req.mode)
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)