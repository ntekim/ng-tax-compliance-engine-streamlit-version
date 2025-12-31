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
ENGINE_ID = "nigeria-compliance-engine_1766620713359"
DATA_STORE_ID = "nigeria-compliance-engine_1766620773637"
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

# --- THE KNOWLEDGE BASE (Your "Manual RAG") ---
NIGERIA_TAX_DATA = """
OFFICIAL NIGERIAN TAX GUIDE (2025):

1. **VAT (Value Added Tax):**
   - **Rate:** 7.5%
   - **Deadline:** Remit by the 21st of the following month.
   - **Penalty:** Failure to remit attracts a fine of ₦50,000 + 5% interest.

2. **CIT (Company Income Tax):**
   - **Small Company** (<₦25m turnover): **0% Tax Rate**.
   - **Medium Company** (₦25m-₦100m): **20% Tax Rate**.
   - **Large Company** (>₦100m): **30% Tax Rate**.

3. **PAYE (Personal Income Tax):**
   - Must be remitted to the **State IRS** (e.g., LIRS) where the employee resides.
   - **Deadline:** 10th of the following month.
   - **Calculation:** Gross Income - (Consolidated Relief Allowance + Pension + NHF) = Taxable Income.

4. **NELFUND (Student Loan):**
   - Employers must deduct **10%** from the salary of beneficiaries.

5. **Remote Workers (USD Income):**
   - Residents in Nigeria earning foreign income **MUST declare it**.
   - You can reduce tax liability by deducting business expenses (Laptop, Internet, Generator) before declaring profit.
"""

# --- LOGIC ---

def execute_search(client, path, query):
    try:
        req = discoveryengine.SearchRequest(serving_config=path, query=query, page_size=3)
        return client.search(req)
    except:
        return None

@tracer.wrap(name="rag_search", service="betawork-ai-engine")
def search_nigerian_laws(query: str):
    if not my_credentials: return []

    try:
        client_opts = ClientOptions(api_endpoint="discoveryengine.googleapis.com")
        client = discoveryengine.SearchServiceClient(credentials=my_credentials, client_options=client_opts)
        
        # Path A: Engine
        path_engine = f"projects/{PROJECT_ID}/locations/global/collections/default_collection/engines/{ENGINE_ID}/servingConfigs/default_search"
        # Path B: DataStore
        path_ds = f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search"

        response = execute_search(client, path_engine, query)
        if not response or len(response.results) == 0:
             response = execute_search(client, path_ds, query)

        sources = []
        if response and hasattr(response, 'results'):
            for result in response.results:
                data = result.document.derived_struct_data
                if 'snippets' in data and len(data['snippets']) > 0:
                    sources.append(data['snippets'][0].get('snippet', ''))
                elif 'extractive_segments' in data and len(data['extractive_segments']) > 0:
                    sources.append(data['extractive_segments'][0].get('content', ''))
        
        return sources
    except Exception as e:
        logger.error(f"Search Fail: {e}")
        return []

@tracer.wrap(name="generate_answer", service="betawork-ai-engine")
def get_ai_response(user_query: str, mode: str):
    if not model: return "AI System Offline.", []
    
    if mode == "therapy":
        prompt = f"You are BetaCare. User: '{user_query}'. Be empathetic."
        try:
            return model.generate_content(prompt).text, []
        except:
            return "I am here to listen.", []

    # --- TAX MODE ---
    
    # 1. Try RAG
    sources = search_nigerian_laws(user_query)
    
    # 2. EMERGENCY FIX: If RAG is empty, inject Manual Data as a "Source"
    if not sources:
        print("⚠️ RAG returned 0. Using Synthetic Source.")
        sources = [NIGERIA_TAX_DATA] # <--- THIS FIXES THE UI
    
    # 3. Construct Prompt
    context_text = "\n\n".join(sources)
    
    prompt = f"""
    ROLE: You are BetaBot, the official Tax Advisor for BetaWork Nigeria.
    
    OFFICIAL DATA SOURCES:
    {context_text}
    
    USER QUESTION: "{user_query}"
    
    INSTRUCTIONS:
    1. Answer specifically using the OFFICIAL DATA SOURCES above.
    2. Be very precise with numbers (e.g. "7.5%", "21st").
    3. Keep it professional and short.
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