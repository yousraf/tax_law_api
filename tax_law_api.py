from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv, find_dotenv

# Load environment variables
#env_path = find_dotenv()
#if not env_path:
    #print("WARNING: .env file not found")
#load_dotenv(env_path)

app = FastAPI(title="Tax Law RAG API")

class TaxQuery(BaseModel):
    query: str

class TaxResponse(BaseModel):
    title: str
    tax_research: str
    tax_citations: str
    draft_client_response: str
    clarifying_questions: str
    confirmation: str

class TaxLawQueryEngine:
    def __init__(self, index_name: str, openai_api_key: str, pinecone_api_key: str):
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key,
        )
        self.embedding_store = PineconeVectorStore(
            index_name=index_name,
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
            model="gpt-4o",
            temperature=0.0
        )

    def generate_response_prompt(self, query: str, context: str) -> str:
        return f"""You are a tax law advisor in Australia. Analyze the following query and provide a structured response with exactly six specific components:

Query: {query}

Context: {context}

Provide your response in the following format, with each section clearly marked by its header:

TITLE:
[Provide a concise title summarizing the tax question or topic in 7 words or less]

TAX_RESEARCH:
[Provide detailed tax research and analysis with references to legislation sections, ATO rulings and interpretations, and latest tax information. Present as bullet points.]

TAX_CITATIONS:
[List relevant tax legislation sections and ATO rules with their titles and numbers]

DRAFT_CLIENT_RESPONSE:
[Provide a professional email response in this format:

Dear [Client Name],

# [Title]

I refer to your query regarding [brief description].

## Background
[Facts as bullet points]

## Summary of Advice
[Concise summary]

## Detailed Analysis
[Analysis with references]

## Additional Comments
[Standard comments]

## Scope and Limitations
[Standard limitations]

Yours sincerely,
[Your Name]]

CLARIFYING_QUESTIONS:
[List specific questions needed to improve the tax analysis. If none needed, state "No further questions."]

CONFIRMATION:
[Provide a short confirmation message about whether the research and draft are complete]

Ensure each section is clearly marked with its header and separated by blank lines."""

    def parse_response(self, response: str) -> TaxResponse:
        sections = {
            "TITLE": "",
            "TAX_RESEARCH": "",
            "TAX_CITATIONS": "",
            "DRAFT_CLIENT_RESPONSE": "",
            "CLARIFYING_QUESTIONS": "",
            "CONFIRMATION": ""
        }
        
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.upper() in sections:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.upper()
                current_content = []
            elif current_section and line:
                current_content.append(line)
                
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return TaxResponse(
            title=sections["TITLE"],
            tax_research=sections["TAX_RESEARCH"],
            tax_citations=sections["TAX_CITATIONS"],
            draft_client_response=sections["DRAFT_CLIENT_RESPONSE"],
            clarifying_questions=sections["CLARIFYING_QUESTIONS"],
            confirmation=sections["CONFIRMATION"]
        )

    def answer_question(self, query: str) -> TaxResponse:
        context = self.query_engine.retrieve_context(query)
        prompt = self.generate_response_prompt(query, context)
        response = self.llm.invoke(prompt)
        return self.parse_response(response.content)

# Global RAG instance
rag_instance = None

@app.on_event("startup")
async def startup_event():
    global rag_instance
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "taxlawlegato")
    
    if not all([openai_api_key, pinecone_api_key]):
        raise RuntimeError("Missing required environment variables")
    
    print("Initializing RAG system...")
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
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
