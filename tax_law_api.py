from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone, PineconeVectorStore
import pinecone
import json
from dotenv import load_dotenv, find_dotenv

# Load environment variables
env_path = find_dotenv()
if not env_path:
    print("WARNING: .env file not found")
load_dotenv(env_path)

app = FastAPI(title="Tax Law RAG API")

class TaxQuery(BaseModel):
    query: str

class TaxResponse(BaseModel):
    background: str
    summary_of_advice: str
    detailed_analysis: str
    citations: List[str]

class TaxLawQueryEngine:
    def __init__(self, index_name: str, openai_api_key: str, pinecone_api_key: str):
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key,
        )
        self.embedding_store = PineconeVectorStore(
            index_name="taxlawlegato",
            embedding=embedding_model,
            pinecone_api_key=pinecone_api_key
        )

    def retrieve_context(self, query: str, k: int = 4) -> str:
        results = self.embedding_store.similarity_search(query, k=k)
        context = ""
        for result in results:
            context += f"\nSection: {result.metadata['full_reference']}\n"
            context += f"{result.page_content}\n"
        return context

class TaxLawRAG:
    def __init__(self, query_engine: TaxLawQueryEngine, openai_api_key: str):
        self.query_engine = query_engine
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4",
            temperature=0.0
        )

    def generate_analysis_prompt(self, query: str, context: str) -> str:
        return f"""You are a tax law advisor in Australia. Analyze the following query and context:

Context:
{context}

Question:
{query}

Provide a comprehensive analysis focusing on:
- Detailed legal reasoning
- Specific tax law considerations
- Key assumptions
- Areas needing clarification

Format your response as a JSON object with these exact fields:
{{
    "Background": "Key facts from the query",
    "SummaryOfAdvice": "Concise summary of tax advice",
    "DetailedAnalysis": "In-depth legal analysis with citations",
    "Citations": ["List of relevant legal references"]
}}"""

    def get_analysis(self, query: str) -> dict:
        context = self.query_engine.retrieve_context(query)
        analysis_prompt = self.generate_analysis_prompt(query, context)
        response = self.llm.invoke(analysis_prompt)
        
        response_content = response.content.strip()
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[7:-3].strip()
        
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as valid JSON")

# Global RAG instance
rag_instance = None

@app.on_event("startup")
async def startup_event():
    global rag_instance
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "taxlawlegato")
    
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY environment variable is not set")
    
    print("Initializing RAG system...")
    query_engine = TaxLawQueryEngine(
        index_name=pinecone_index_name,
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key
    )
    rag_instance = TaxLawRAG(query_engine, openai_api_key)
    print("RAG system initialized successfully")

@app.post("/analyze")
async def analyze_tax_query(query: TaxQuery) -> TaxResponse:
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    try:
        analysis = rag_instance.get_analysis(query.query)
        return TaxResponse(
            background=analysis["Background"],
            summary_of_advice=analysis["SummaryOfAdvice"],
            detailed_analysis=analysis["DetailedAnalysis"],
            citations=analysis["Citations"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/background/{query_id}")
async def get_background(query_id: str):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    try:
        analysis = rag_instance.get_analysis(query_id)
        return {"background": analysis["Background"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{query_id}")
async def get_summary(query_id: str):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    try:
        analysis = rag_instance.get_analysis(query_id)
        return {"summary_of_advice": analysis["SummaryOfAdvice"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{query_id}")
async def get_detailed_analysis(query_id: str):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    try:
        analysis = rag_instance.get_analysis(query_id)
        return {"detailed_analysis": analysis["DetailedAnalysis"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/citations/{query_id}")
async def get_citations(query_id: str):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    try:
        analysis = rag_instance.get_analysis(query_id)
        return {"citations": analysis["Citations"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
