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
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betabot")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "silver-impulse-481722-v5")
# We will use BOTH IDs to be safe
ENGINE_ID = "nigeria-compliance-engine_1766620713359" # App ID
DATA_STORE_ID = "nigeria-compliance-engine_1766620773637" # Data Store ID
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
CREDENTIALS_FILE = os.path.abspath("gcp_key.json")

# --- AUTH SETUP ---
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

# --- VERTEX INIT ---
model = None
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro")
except Exception as e:
    logger.error(f"Vertex Init Failed: {e}")

# --- DATADOG DUMMY ---
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

# --- DATA MODELS ---
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

# --- HARDCODED FALLBACK KNOWLEDGE (Safety Net) ---
NIGERIA_TAX_CHEAT_SHEET = """
1. **VAT (Value Added Tax):** Rate is 7.5%. Remit to FIRS by the 21st of the following month.
2. **CIT (Company Income Tax):** 
   - Small Co (<25m turnover): 0% rate.
   - Medium Co (25m-100m): 20% rate.
   - Large Co (>100m): 30% rate.
   - Due 6 months after financial year end.
3. **PAYE (Personal Income Tax):** Remit to State IRS (e.g. LIRS) by the 10th. Based on graduated scale (7% to 24%).
4. **WHT (Withholding Tax):** Usually 5% or 10% depending on transaction.
5. **NELFUND:** Employers must deduct 10% for Student Loan repayment if applicable.
"""

# --- LOGIC ---

def execute_search(client, path, query):
    """Helper to try a specific path"""
    try:
        req = discoveryengine.SearchRequest(serving_config=path, query=query, page_size=3)
        return client.search(req)
    except Exception as e:
        logger.error(f"Path failed: {path} | Error: {e}")
        return None

@tracer.wrap(name="rag_search", service="betawork-ai-engine")
def search_nigerian_laws(query: str):
    if not my_credentials: return []

    try:
        client_opts = ClientOptions(api_endpoint="discoveryengine.googleapis.com")
        client = discoveryengine.SearchServiceClient(credentials=my_credentials, client_options=client_opts)
        
        # STRATEGY: Try Engine ID first, then Data Store ID
        
        # Path A: Engine (App) Path
        path_engine = f"projects/{PROJECT_ID}/locations/global/collections/default_collection/engines/{ENGINE_ID}/servingConfigs/default_search"
        
        # Path B: Data Store Path
        path_ds = f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search"

        print(f"🔎 Attempting Search for: '{query}'")
        
        # Try A
        response = execute_search(client, path_engine, query)
        
        # If A fails or returns 0 results, Try B
        if not response or len(response.results) == 0:
             print("⚠️ Engine path yielded 0 results. Trying Data Store path...")
             response = execute_search(client, path_ds, query)

        sources = []
        if response and hasattr(response, 'results'):
            print(f"✅ Final Results: {len(response.results)}")
            for result in response.results:
                data = result.document.derived_struct_data
                if 'snippets' in data and len(data['snippets']) > 0:
                    sources.append(data['snippets'][0].get('snippet', ''))
                elif 'extractive_segments' in data and len(data['extractive_segments']) > 0:
                    sources.append(data['extractive_segments'][0].get('content', ''))
        
        return sources
    except Exception as e:
        logger.error(f"Search Critical Fail: {e}")
        return []

@tracer.wrap(name="generate_answer", service="betawork-ai-engine")
def get_ai_response(user_query: str, mode: str):
    if not model: return "AI System Offline.", []
    
    sources = []
    
    if mode == "therapy":
        prompt = f"You are BetaCare, a therapist. User: '{user_query}'. Be empathetic."
    else:
        # 1. RAG Search
        sources = search_nigerian_laws(user_query)
        
        # 2. Context Construction
        if sources:
            context_text = "\n\n".join([f"SOURCE {i+1}: {s}" for i, s in enumerate(sources)])
            context_source = "OFFICIAL DOCUMENTS (Vertex AI)"
        else:
            # FALLBACK INJECTION
            context_text = NIGERIA_TAX_CHEAT_SHEET
            context_source = "INTERNAL KNOWLEDGE BASE (Fallback)"
        
        prompt = f"""
        ROLE: You are BetaBot, a Tax Compliance Expert for Nigeria.
        
        KNOWLEDGE SOURCE ({context_source}):
        {context_text}
        
        USER QUESTION: "{user_query}"
        
        INSTRUCTIONS:
        1. Answer the question using the KNOWLEDGE SOURCE above.
        2. Be specific (mention rates, deadlines, and acronyms like FIRS/LIRS).
        3. Keep it under 150 words.
        4. Use bullet points.
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
    
    answer_text, source_list = get_ai_response(final_query, req.mode)
    
    return {"answer": answer_text, "sources": source_list}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)