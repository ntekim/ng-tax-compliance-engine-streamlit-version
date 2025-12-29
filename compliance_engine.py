import os
import vertexai
from ddtrace import tracer
from google.cloud import bigquery, discoveryengine_v1beta as discoveryengine
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "silver-impulse-481722-v5"
DATA_STORE_ID = "nigeria-compliance-engine_1766620773637" # e.g., 'nigeria-laws_123'
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-pro")

def search_nigerian_laws(query):
    """Manually search your PDFs in Vertex AI Search"""
    with tracer.trace("law_search", service="reguguard-3") as span:
        client = discoveryengine.SearchServiceClient()
        serving_config = f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search"
        
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=3, # Get the top 3 relevant sections
        )
        
        response = client.search(request)
        
        # Extract snippets from the PDFs
        context = ""
        for result in response.results:
            snippet = result.document.derived_struct_data.get('snippets', [{}])[0].get('snippet', "")
            context += f"\n--- Law Segment ---\n{snippet}\n"
        
        span.set_tag("search.results_count", len(response.results))
        return context

@tracer.wrap(name="compliance_check", service="reguguard-3")
def get_compliance_response(user_query):
    # 1. Get Economic Context from BigQuery
    gdp_val = "2.5%" 
    # (Your existing BigQuery code here...)

    # 2. STEP 1: Manual Search (RAG)
    legal_context = search_nigerian_laws(user_query)

    # 3. STEP 2: Gemini Reasoning
    try:
        with tracer.trace("legal_reasoning") as span:
            prompt = f"""
            You are a Nigerian Regulatory Advisor. 
            
            ECONOMICS: Nigeria GDP is {gdp_val}.
            LEGAL CONTEXT FROM STATUTES: {legal_context}
            
            USER QUESTION: {user_query}
            
            INSTRUCTION: Use ONLY the provided legal context to answer. 
            Explain it simply for a small business owner. 
            """
            
            response = model.generate_content(prompt)
            
            span.set_tag("llm.model", "gemini-2.5-pro")
            return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"