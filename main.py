import os
import json
import base64
import logging
import uvicorn
import vertexai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import Optional
from google.cloud import discoveryengine_v1beta as discoveryengine
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

# --- CONFIGURATION (Same as before) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betabot")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "silver-impulse-481722-v5")
DATA_STORE_ID = os.getenv("GCP_DATA_STORE_ID", "nigeria-compliance-engine_1766620773637")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
CREDENTIALS_FILE = os.path.abspath("gcp_key.json")

# --- AUTH SETUP (Same as before) ---
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

# --- INIT VERTEX AI (Same as before) ---
model = None
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro")
except Exception as e:
    logger.error(f"Vertex Init Failed: {e}")

# --- DATADOG (Same as before) ---
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

# --- DATA MODELS (Same as before) ---
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

# --- IMPROVED LOGIC ---

@tracer.wrap(name="rag_search", service="betawork-ai-engine")
def search_nigerian_laws(query: str):
    if not my_credentials: return ""

    try:
        client = discoveryengine.SearchServiceClient(credentials=my_credentials)
        
        # FIX 1: OPTIMIZE QUERY
        # If user types "Business Policy", we send "Business Policy Nigeria Tax Law"
        # This helps the Search Engine find relevant docs instead of returning nothing.
        enhanced_query = f"{query} regarding Nigeria Tax and Finance Act"
        ENGINE_ID = "nigeria-compliance-engine_1766620713359"
        serving_config = client.serving_config_path(
            project=PROJECT_ID,
            location="global",
            collection="default_collection",
            data_store=ENGINE_ID,
            serving_config="default_search",
        )
        
        req = discoveryengine.SearchRequest(
            serving_config=serving_config, 
            query=enhanced_query, 
            page_size=3
        )
        
        response = client.search(req)
        
        context = ""
        for result in response.results:
            data = result.document.derived_struct_data
            if 'snippets' in data and len(data['snippets']) > 0:
                context += f"\n--- RELEVANT NIGERIAN LAW ---\n{data['snippets'][0].get('snippet', '')}\n"
        return context
    except Exception as e:
        logger.error(f"Search API Error: {e}")
        return ""

@tracer.wrap(name="generate_answer", service="betawork-ai-engine")
def get_ai_response(user_query: str, mode: str):
    if not model: return "AI System Offline."
    
    if mode == "therapy":
        # THERAPY PROMPT (Unchanged)
        prompt = f"You are BetaCare, a therapist. User: '{user_query}'. Be empathetic."
    else:
        # TAX PROMPT (HEAVILY IMPROVED)
        legal_context = search_nigerian_laws(user_query)
        
        prompt = f"""
        ROLE: You are BetaBot, a specialized Tax Compliance Consultant for Nigerian SMEs.
        
        CONTEXT FROM NIGERIAN DOCUMENTS (RAG):
        {legal_context if legal_context else "No specific document section found. Rely on your internal knowledge of FIRS, LIRS, and CAMA 2020."}
        
        USER QUESTION: "{user_query}"
        
        RULES FOR ANSWERING:
        1. **STRICTLY NIGERIAN CONTEXT:** Never mention "IRS" (US). Only mention "FIRS" (Federal) or "State IRS" (LIRS, etc).
        2. **SIMPLICITY:** Explain it like you are talking to a 25-year-old business owner, not a lawyer. Avoid jargon.
        3. **FORMATTING:** Use short paragraphs and Bullet Points.
        4. **BE CONCISE:** Keep the answer under 200 words.
        5. **ACTIONABLE:** Tell them exactly what to do next (e.g., "File via TaxPro Max" or "Log this as an expense").
        """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Thinking Error: {e}"

# --- ENDPOINTS (Same as before) ---
@app.get("/")
def root(): return {"status": "running"}

@app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    final_query = req.query or req.question
    if not final_query: raise HTTPException(status_code=400, detail="Query required")
    answer = get_ai_response(final_query, req.mode)
    return {"answer": answer}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)