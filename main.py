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
            # Convert to Dict
            data = MessageToDict(result.document._pb)
            derived = data.get("derivedStructData", {})
            
            # THE FIX: Check 'extractive_answers'
            if "extractive_answers" in derived:
                for answer in derived["extractive_answers"]:
                    # Grab the 'content' field
                    text = answer.get("content", "")
                    if text:
                        sources.append(text)
            
            # Fallback (Keep these just in case)
            elif "snippets" in derived:
                for s in derived["snippets"]:
                    sources.append(s.get("snippet", ""))
            
            elif "extractive_segments" in derived:
                for s in derived["extractive_segments"]:
                    sources.append(s.get("content", ""))

        return sources[:3] # Return top 3
        
    except Exception as e:
        logger.error(f"Search API Error: {e}")
        return []

tracer.wrap(name="generate_answer", service="betawork-ai-engine")
def get_ai_response(user_query: str, mode: str):
    if not model: return "AI System Offline.", []
    
    # 1. Therapy Mode (Unchanged)
    if mode == "therapy":
        prompt = f"You are BetaCare, a therapist. User: '{user_query}'. Be empathetic."
        try:
            return model.generate_content(prompt).text, []
        except:
            return "I am listening.", []

    # 2. Business/Tax Mode
    sources = search_nigerian_laws(user_query)
    econ_data = get_economic_context()
    
    rag_text = ""
    if sources:
        rag_text = "OFFICIAL DOCUMENTS:\n" + "\n".join(sources)
    
    # THE HYBRID PROMPT
    prompt = f"""
    ROLE: You are 'BetaBot', a Strategic Business Advisor for Nigerian SMEs.
    You are an expert in Tax Law, Business Strategy, and Financial Growth.

    ECONOMIC CONTEXT:
    {econ_data}
    {rag_text}

    USER QUESTION: "{user_query}"

    INSTRUCTIONS:
    1. **CLASSIFY:** Is this a Tax/Legal question or a General Business question?
    
    2. **IF TAX/LEGAL:**
       - Be precise. Cite the 'OFFICIAL DOCUMENTS' if available.
       - Quote rates (7.5% VAT) and deadlines (21st).
       - Keep it strict and compliant.

    3. **IF GENERAL BUSINESS (e.g. "How to scale?", "Marketing tips"):**
       - Be creative and strategic.
       - Use the 'ECONOMIC CONTEXT' (Population/GDP) to give localized advice (e.g. "Given Nigeria's population of 200M...").
       - Do NOT force tax laws into the answer unless relevant.

    4. **TONE:** Professional, concise, and encouraging. Max 150 words.
    """

    try:
        response = model.generate_content(prompt)
        return response.text, sources, econ_data
    except Exception as e:
        return f"Thinking Error: {e}", []
    
# --- ENDPOINTS ---
@app.post("/ask", response_model=APIResponse)
async def ask_endpoint(req: QueryRequest):
    final_query = req.query or req.question
    if not final_query: raise HTTPException(status_code=400, detail="Query required")
    
    # Unpack 3 values now
    answer, sources, econ_data = get_ai_response(final_query, req.mode)
    
    return {
        "answer": answer,
        "sources": sources,
        "economic_data": econ_data # Send to Frontend
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)