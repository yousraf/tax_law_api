# tax_law_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone

import json
from dotenv import load_dotenv, find_dotenv


app = FastAPI(title="Tax Law RAG API")

class TaxQuery(BaseModel):
    query: str

class TaxLawQueryEngine:
    def __init__(self, index_name: str, openai_api_key: str, pinecone_api_key: str):
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key,
        )

        # Initialize Pinecone
        #pc = Pinecone(api_key=pinecone_api_key)
        #self.embedding_store = PineconeVectorStore(
           # index_name="taxlawlegato",
           #embedding=embedding_model
        #)

        self.embedding_store = PineconeVectorStore(index_name="taxlawlegato", embedding=embedding_model, pinecone_api_key=pinecone_api_key)


    def retrieve_context(self, query: str, k: int = 4) -> str:
        results = self.embedding_store.similarity_search(query, k=k)
        context = ""
        for result in results:
            context += f"\nSection: {result.metadata['full_reference']}\n"
            context += f"{result.page_content}\n"
        return context

# Rest of the code remains the same
class TaxLawRAG:
    def __init__(self, query_engine: TaxLawQueryEngine, openai_api_key: str):
        self.query_engine = query_engine
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4o",
            temperature=0.0
        )

    def generate_answer_prompt(self, query: str, context: str) -> str:
        return f"""You are a tax law advisor in Australia. Your task is to research and analyse the tax context for the following query and provide relevant links for further reference based on the metadata:

Context:
{context}

Question:
{query}

Provide a comprehensive, step-by-step analysis of the tax implications. Focus on:

    Detailed legal reasoning
    Specific tax law considerations
    Identifying key assumptions
    Suggesting areas that need further clarification by asking targeted questions as required.

Ensure your response includes:

    A clear, structured analysis.
    Relevant links to legal sources from '''https://www.ato.gov.au/''' related to the context and query.
    An easily convertible format to JSON with fields such as analysis, assumptions, clarifications_needed, and references.

Respond concisely while ensuring accuracy and relevance."""

    def generate_json_prompt(self, analysis: str) -> str:
        return f"""Convert the following analysis into a structured JSON response:

Analysis:
{analysis}

Respond ONLY with a valid JSON object in this format:
{{
    "Background": "Summarise key facts provided by the client.",
    "SummaryOfAdvice": "Succinctly summarise the key tax advice.",
    "DetailedAnalysis": "Provide detailed legal analysis, citing legislation and ATO rulings.",
    "Citations": ["List references to legislation, rulings, or materials."]
}}"""

    def format_draft_from_json(self, json_response: dict) -> str:
        return f"""Dear [Client Name],

Tax Law Advice

I refer to your query regarding tax implications.

Background
You have provided the following by way of background facts:
- {json_response.get("Background", "No background provided.")}

Please note that the accuracy and completeness of the information provided by you is critical to the advice being provided. Kindly inform me if any of the details are incomplete or inaccurate.

Summary of Advice
{json_response.get("SummaryOfAdvice", "No advice summary provided.")}

Detailed Analysis
{json_response.get("DetailedAnalysis", "No detailed analysis provided.")}

Citations:
{", ".join(json_response.get("Citations", []))}

Additional Comments
While this advice is limited to the tax issues specified, other tax considerations may apply based on the facts provided. Should you wish for further analysis on additional aspects, we are happy to assist upon request.

Scope and Limitations
This advice is based solely on the facts provided above. Any changes to the facts may affect the conclusions reached. Please notify us if any details need to be corrected or updated. The content of this advice reflects the current state of tax law as of the date issued and may need revision if relevant legislation or interpretations change.

If you have any further questions or require clarification, please do not hesitate to contact me.

Yours sincerely,
[Your Name]"""

    def answer_question(self, query: str) -> str:
        context = self.query_engine.retrieve_context(query)
        answer_prompt = self.generate_answer_prompt(query, context)
        initial_analysis = self.llm.invoke(answer_prompt)
        
        json_prompt = self.generate_json_prompt(initial_analysis.content)
        json_response = self.llm.invoke(json_prompt)
        
        response_content = json_response.content.strip()
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[7:-3].strip()
        
        try:
            json_data = json.loads(response_content)
        except json.JSONDecodeError:
            raise ValueError("The response from the LLM could not be parsed as valid JSON.")
        
        draft_prompt = self.format_draft_from_json(json_data)
        draft_response = self.llm.invoke(draft_prompt)
        
        return draft_response.content

# Global RAG instance
rag_instance = None

@app.on_event("startup")
async def startup_event():
    global rag_instance
    
    # Get environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "taxlawlegato")
    
    # Check if required environment variables are set
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    if not pinecone_api_key: 
        raise RuntimeError("PINECONE_API_KEY environment variable is not set")
    
    print("Initializing RAG system...")
    print(f"Using Pinecone index: {pinecone_index_name}")
    
    query_engine = TaxLawQueryEngine(
        index_name=pinecone_index_name,
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key
    )
    rag_instance = TaxLawRAG(query_engine, openai_api_key)
    print("RAG system initialized successfully")


@app.post("/query")
async def query_tax_law(query: TaxQuery):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    try:
        response = rag_instance.answer_question(query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/")
async def root():
    return {
        "message": "Tax Law API is running",
        "status": "healthy",
        "available_endpoints": [
            {
                "path": "/health",
                "method": "GET",
                "description": "Check API status"
            },
            {
                "path": "/query",
                "method": "POST",
                "description": "Submit tax law query",
                "request_body": {
                    "query": "string - Your tax law question"
                }
            }
        ],
        "documentation": "Explore API endpoints and functionalities"
    }
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run("tax_law_api:app", host="0.0.0.0", port=port, reload=True)
