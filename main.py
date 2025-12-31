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
from google.protobuf.json_format import MessageToDict # HELPER FOR PARSING

# --- 1. CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betabot")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "silver-impulse-481722-v5")
# Using the DATA STORE ID from your screenshot (Ending in 73637)
DATA_STORE_ID = os.getenv("GCP_DATA_STORE_ID", "nigeria-compliance-engine_1766620773637")
LOCATION = "global" # FORCE GLOBAL based on your screenshot
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
    # Vertex AI (Gemini) - Keep in us-central1 for Model availability
    vertexai.init(project=PROJECT_ID, location="us-central1") 
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
    economic_data: str

# --- 6. LOGIC ---

def get_economic_context():
    if not bq_client: return ""
    try:
        query = """
            SELECT indicator_name, value, year
            FROM `bigquery-public-data.world_bank_wdi.indicators_data`
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
def search_nigerian_laws(query: str):
    """Robust Search that checks all possible result fields."""
    if not my_credentials: return []

    try:
        # Use Default Client (Global Endpoint)
        client = discoveryengine.SearchServiceClient(credentials=my_credentials)
        
        # Path to DATA STORE (Global)
        serving_config = (
            f"projects/{PROJECT_ID}/locations/global/collections/default_collection/"
            f"dataStores/{DATA_STORE_ID}/servingConfigs/default_search"
        )
        
        # Add query expansion to find related terms
        req = discoveryengine.SearchRequest(
            serving_config=serving_config, 
            query=query, 
            page_size=3,
            query_expansion_spec={"condition": "AUTO"},
            spell_correction_spec={"mode": "AUTO"}
        )
        
        response = client.search(req)
        print(f"🔎 Results Found: {len(response.results)}")
        
        sources = []
        for result in response.results:
            # Convert Protobuf to Dict to make access safer
            data = MessageToDict(result.document._pb)
            derived = data.get("derivedStructData", {})
            
            # ATTEMPT 1: Snippets (Standard)
            if "snippets" in derived:
                for s in derived["snippets"]:
                    sources.append(s.get("snippet", ""))
            
            # ATTEMPT 2: Extractive Segments (PDFs often use this)
            elif "extractiveSegments" in derived:
                for s in derived["extractiveSegments"]:
                    sources.append(s.get("content", ""))
            
            # ATTEMPT 3: Raw Text (Fallback)
            elif "structData" in data and "text" in data["structData"]:
                sources.append(data["structData"]["text"][:500]) # First 500 chars

        # Filter empty strings
        clean_sources = [s for s in sources if s.strip()]
        return clean_sources[:3] # Return top 3
        
    except Exception as e:
        logger.error(f"Search API Error: {e}")
        return []

@tracer.wrap(name="generate_answer", service="betawork-ai-engine")
def get_ai_response(user_query: str, mode: str):
    if not model: return "AI System Offline.", [], ""
    
    if mode == "therapy":
        prompt = f"You are BetaCare, a therapist. User: '{user_query}'. Be empathetic."
        try:
            return model.generate_content(prompt).text, [], ""
        except:
            return "I am listening.", [], ""

    # Mode 2: Business/Tax
    
    # 1. Fetch Context
    sources = search_nigerian_laws(user_query)
    econ_data = get_economic_context()
    
    # 2. Build Prompt
    rag_text = ""
    if sources:
        rag_text = "OFFICIAL DOCUMENTS FOUND:\n" + "\n---\n".join(sources)
    else:
        # If still empty, use fallback logic but DO NOT mock sources for the UI
        rag_text = "No specific document found in Vector DB. Use general knowledge."
    
    prompt = f"""
    You are BetaBot, a Nigerian Tax Advisor.
    
    ECONOMIC CONTEXT:
    {econ_data}
    
    {rag_text}
    
    USER QUESTION: "{user_query}"
    
    INSTRUCTIONS:
    1. Answer based on the documents above if they exist.
    2. Be professional and concise.
    """

    try:
        response = model.generate_content(prompt)
        return response.text, sources, econ_data
    except Exception as e:
        return f"Thinking Error: {e}", [], ""

# --- ENDPOINTS ---
@app.get("/")
def root(): return {"status": "running"}

@app.post("/ask", response_model=APIResponse)
async def ask_endpoint(req: QueryRequest):
    final_query = req.query or req.question
    if not final_query: raise HTTPException(status_code=400, detail="Query required")
    
    answer, sources, econ_data = get_ai_response(final_query, req.mode)
    return {"answer": answer, "sources": sources, "economic_data": econ_data}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)