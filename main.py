import os
import logging
import vertexai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import discoveryengine_v1beta as discoveryengine
from vertexai.generative_models import GenerativeModel

# --- CONFIG ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "silver-impulse-481722-v5")
DATA_STORE_ID = os.getenv("GCP_DATA_STORE_ID", "nigeria-compliance-engine_1766620773637")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("betabot")

# --- DATADOG SAFETY ---
try:
    from ddtrace import tracer, patch_all
    from ddtrace.contrib.fastapi import TraceMiddleware
    patch_all()
    DD_ENABLED = True
except (ImportError, Exception):
    DD_ENABLED = False
    class DummyTracer:
        def wrap(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
    tracer = DummyTracer()

# --- GCP SETUP ---
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro")
except Exception as e:
    model = None
    logger.error(f"Vertex AI Init Failed: {e}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if DD_ENABLED:
    app.add_middleware(TraceMiddleware, service="betawork-ai-engine")

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    query: str
    mode: str = "tax" # 'tax' or 'therapy'

# --- LOGIC ---
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
    if not model: return "AI Service Unavailable."

    if mode == "therapy":
        # THERAPY PERSONA
        prompt = f"""
        You are 'BetaCare', an empathetic workplace counselor for Nigerian professionals.
        
        USER SAYS: "{user_query}"
        
        INSTRUCTIONS:
        1. Listen actively and validate their feelings (burnout, stress, financial anxiety).
        2. Keep responses warm, short, and conversational (like a kind colleague).
        3. Do NOT give medical advice. If they mention self-harm or severe issues, urged them to use the 'Book Human Expert' button.
        4. Ask one gentle follow-up question to help them vent.
        """
    else:
        # TAX PERSONA (Default)
        legal_context = search_nigerian_laws(user_query)
        prompt = f"""
        You are 'BetaBot', a Nigerian Tax Compliance Advisor.
        
        LEGAL CONTEXT (RAG): {legal_context}
        USER QUESTION: "{user_query}"
        
        INSTRUCTIONS:
        1. Answer based on the Legal Context if available.
        2. Be professional and concise.
        """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "I'm having trouble thinking right now."

@app.post("/ask")
async def ask_endpoint(req: QueryRequest):
    answer = get_ai_response(req.query, req.mode)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)