import os
import json
import base64
import logging
import uvicorn
import vertexai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google.cloud import discoveryengine_v1beta as discoveryengine
from vertexai.generative_models import GenerativeModel

# --- 1. CRITICAL AUTH SETUP (MUST BE AT TOP) ---
# This converts the Render Environment Variable back into a file
if os.getenv("GCP_CREDENTIALS_BASE64"):
    try:
        print("🔐 Found Base64 Credentials. Decoding...")
        decoded_key = base64.b64decode(os.getenv("GCP_CREDENTIALS_BASE64"))
        with open("gcp_key.json", "w") as f:
            f.write(decoded_key.decode("utf-8"))
        
        # Set the Env Var that Google Libraries look for
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gcp_key.json")
        print(f"✅ Auth File Created at: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
    except Exception as e:
        print(f"❌ FATAL: Failed to decode credentials: {e}")

# --- 2. CONFIGURATION ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "silver-impulse-481722-v5")
DATA_STORE_ID = os.getenv("GCP_DATA_STORE_ID", "nigeria-compliance-engine_1766620773637")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betabot")

# --- 3. DATADOG DUMMY WRAPPER ---
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

# --- 4. INIT VERTEX AI ---
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro")
except Exception as e:
    model = None
    logger.error(f"Vertex Init Error (Check Credentials): {e}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if DD_ENABLED:
    app.add_middleware(TraceMiddleware, service="betawork-ai-engine")

# --- 5. DATA MODEL (Flexible) ---
class QueryRequest(BaseModel):
    # Allow 'query' OR 'question' to prevent 422 errors
    query: str = Field(alias="question", default=None) 
    mode: str = "tax"

    # Custom validator to handle if user sends 'query' directly
    def __init__(self, **data):
        # If user sent 'query', use it. If 'question', map it to query.
        if 'question' in data:
            data['query'] = data['question']
        super().__init__(**data)

# --- 6. LOGIC ---
@tracer.wrap(name="rag_search", service="betawork-ai-engine")
def search_nigerian_laws(query: str):
    try:
        client = discoveryengine.SearchServiceClient()
        serving_config = f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search"
        req = discoveryengine.SearchRequest(serving_config=serving_config, query=query, page_size=3)
        response = client.search(req)
        
        context = ""
        for result in response.results:
            data = result.document.derived_struct_data
            if 'snippets' in data and len(data['snippets']) > 0:
                context += f"\n--- LAW SNIPPET ---\n{data['snippets'][0].get('snippet', '')}\n"
        return context
    except Exception as e:
        logger.error(f"Search Failed: {e}")
        return ""

@tracer.wrap(name="generate_answer", service="betawork-ai-engine")
def get_ai_response(user_query: str, mode: str):
    if not model: return "AI System is offline (Auth Error)."
    
    if mode == "therapy":
        prompt = f"You are BetaCare, a workplace therapist. Listen to this user: {user_query}. Be empathetic."
    else:
        legal_context = search_nigerian_laws(user_query)
        prompt = f"You are BetaBot, a Nigerian Tax Advisor.\n\nLEGAL CONTEXT:\n{legal_context}\n\nQUESTION: {user_query}\n\nANSWER:"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Thinking Error: {e}"

@app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Please provide a 'query' or 'question'")
    
    answer = get_ai_response(req.query, req.mode)
    return {"answer": answer}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)