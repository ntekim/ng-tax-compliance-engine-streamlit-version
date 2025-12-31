import os
import json
import base64
import logging
import uvicorn
import vertexai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Optional
from google.cloud import discoveryengine_v1beta as discoveryengine
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account # <--- REQUIRED

# --- 1. CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betabot")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "silver-impulse-481722-v5")
DATA_STORE_ID = os.getenv("GCP_DATA_STORE_ID", "nigeria-compliance-engine_1766620773637")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
CREDENTIALS_FILE = os.path.abspath("gcp_key.json")

# --- 2. AUTH SETUP ---
my_credentials = None

if os.getenv("GCP_CREDENTIALS_BASE64"):
    try:
        print("🔐 Decoding Credentials...")
        decoded_key = base64.b64decode(os.getenv("GCP_CREDENTIALS_BASE64"))
        with open(CREDENTIALS_FILE, "w") as f:
            f.write(decoded_key.decode("utf-8"))
        
        # 1. Set Env Var (For Vertex AI)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
        
        # 2. Load Object Explicitly (For Search Client)
        my_credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
        print("✅ Explicit Credentials Object Created")
    except Exception as e:
        print(f"❌ Auth Error: {e}")

# --- 3. INIT VERTEX AI ---
model = None
try:
    # Vertex AI usually picks up the Env Var fine
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro")
    print("✅ Vertex AI Connected")
except Exception as e:
    logger.error(f"Vertex Init Failed: {e}")

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

# --- 6. LOGIC ---
@tracer.wrap(name="rag_search", service="betawork-ai-engine")
def search_nigerian_laws(query: str):
    if not my_credentials:
        logger.error("Missing Credentials Object")
        return ""

    try:
        # FIX: PASS CREDENTIALS EXPLICITLY
        client = discoveryengine.SearchServiceClient(credentials=my_credentials)
        
        serving_config = f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search"
        
        req = discoveryengine.SearchRequest(
            serving_config=serving_config, 
            query=query, 
            page_size=3
        )
        
        response = client.search(req)
        
        context = ""
        for result in response.results:
            data = result.document.derived_struct_data
            if 'snippets' in data and len(data['snippets']) > 0:
                context += f"\n--- LAW SNIPPET ---\n{data['snippets'][0].get('snippet', '')}\n"
        return context
    except Exception as e:
        logger.error(f"Search API Error: {e}")
        # Return empty string so the user still gets a generic AI answer
        return ""

@tracer.wrap(name="generate_answer", service="betawork-ai-engine")
def get_ai_response(user_query: str, mode: str):
    if not model: return "AI System Offline."
    
    if mode == "therapy":
        prompt = f"You are BetaCare, a therapist. User: '{user_query}'. Be empathetic."
    else:
        legal_context = search_nigerian_laws(user_query)
        prompt = f"You are BetaBot, a Tax Advisor.\n\nCONTEXT:\n{legal_context}\n\nQUESTION: {user_query}\n\nANSWER:"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Thinking Error: {e}"

# --- 7. ENDPOINTS ---
@app.get("/")
def root():
    return {"status": "running"}

@app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    final_query = req.query or req.question
    if not final_query:
        raise HTTPException(status_code=400, detail="Query required")
    
    answer = get_ai_response(final_query, req.mode)
    return {"answer": answer}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)