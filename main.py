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

# --- 1. CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betabot")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "silver-impulse-481722-v5")
# Using the DATA STORE ID from your screenshot (Ending in 73637)
DATA_STORE_ID = os.getenv("GCP_DATA_STORE_ID", "nigeria-compliance-engine_1766620773637")
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
        print(f"❌ Auth Error: {e}")

# --- 3. VERTEX INIT ---
model = None
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro")
except Exception as e:
    logger.error(f"Vertex Init Failed: {e}")

# --- 4. DATADOG ---
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

# Response Model for UI
class APIResponse(BaseModel):
    answer: str
    sources: List[str]

# --- 6. LOGIC ---

@tracer.wrap(name="rag_search", service="betawork-ai-engine")
def search_nigerian_laws(query: str):
    """Returns List of Snippets"""
    if not my_credentials: return []

    try:
        client_options = ClientOptions(api_endpoint="discoveryengine.googleapis.com")
        client = discoveryengine.SearchServiceClient(
            credentials=my_credentials, 
            client_options=client_options
        )
        
        # Enhanced Query for better RAG hits
        enhanced_query = f"{query} regarding Nigeria Tax Law"
        
        serving_config = (
            f"projects/{PROJECT_ID}/locations/global/collections/default_collection/"
            f"dataStores/{DATA_STORE_ID}/servingConfigs/default_search"
        )
        
        req = discoveryengine.SearchRequest(
            serving_config=serving_config, 
            query=enhanced_query, 
            page_size=3
        )
        
        response = client.search(req)
        print(f"🔎 Results Found: {len(response.results)}")
        
        sources = []
        for result in response.results:
            data = result.document.derived_struct_data
            if 'snippets' in data and len(data['snippets']) > 0:
                sources.append(data['snippets'][0].get('snippet', ''))
            elif 'extractive_segments' in data and len(data['extractive_segments']) > 0:
                sources.append(data['extractive_segments'][0].get('content', ''))
                
        return sources
    except Exception as e:
        logger.error(f"Search API Error: {e}")
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
        context_text = "\n\n".join([f"SOURCE {i+1}: {s}" for i, s in enumerate(sources)])
        if not context_text:
            context_text = "No specific document found. Use general knowledge of FIRS, LIRS, and Nigerian Finance Acts."
        
        prompt = f"""
        ROLE: You are BetaBot, a specialized Tax Compliance Consultant for Nigerian SMEs.
        
        CONTEXT FROM NIGERIAN DOCUMENTS:
        {context_text}
        
        USER QUESTION: "{user_query}"
        
        RULES:
        1. **STRICTLY NIGERIAN CONTEXT:** Never mention "IRS" (US). Only mention "FIRS" (Federal) or "State IRS".
        2. **SIMPLICITY:** Explain it like you are talking to a business owner.
        3. **FORMATTING:** Use short paragraphs and Bullet Points.
        4. **ACTIONABLE:** Tell them exactly what to do next.
        """

    try:
        response = model.generate_content(prompt)
        return response.text, sources
    except Exception as e:
        return f"Thinking Error: {e}", []

# --- 7. ENDPOINTS ---
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