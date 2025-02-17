from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import re
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
#env_path = find_dotenv()
#if not env_path:
    #print("WARNING: .env file not found")
#load_dotenv(env_path)

app = FastAPI(title="Tax Law RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class TaxQuery(BaseModel):
    query: str
    promptTitle: Optional[str] = None
    promptResearch: Optional[str] = None
    promptCitations: Optional[str] = "Format each citation exactly as:\n\"Citation Name | Citation URL\" (one per line)\nExample: \"ITAA 1997 26-47 | https://www.ato.gov.au/law/view/document?docid=PAC/19970038/26-47\""
    promptClientResponse: Optional[str] = None
    promptClarifyingQuestions: Optional[str] = None
    promptConfirmation: Optional[str] = None

class Citation(BaseModel):
    citations_name: str
    citation_url: str

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

    def generate_response_prompt(self, query: str, context: str, prompt_params: Dict[str, str]) -> str:
        base_prompt = f"""You are a tax law advisor in Australia. Analyze this query and provide exactly six responses.

Query: {query}

Context: {context}

Respond in exactly this format with these exact section headers:

[TITLE]
{prompt_params.get('promptTitle', '')}

[TAX_RESEARCH]
{prompt_params.get('promptResearch', '')}

[TAX_CITATIONS] 
{prompt_params.get('promptCitations', 'Format each citation exactly as:\n"Citation Name | Citation URL" (one per line)\nExample: "ITAA 1997 26-47 | https://www.ato.gov.au/law/view/document?docid=PAC/19970038/26-47"')}

[DRAFT_CLIENT_RESPONSE]
{prompt_params.get('promptClientResponse', '')}

[CLARIFYING_QUESTIONS]
{prompt_params.get('promptClarifyingQuestions', '')}

[CONFIRMATION]
{prompt_params.get('promptConfirmation', '')}

Important: Use the exact section headers shown in [brackets] above. Each section must start with its header on a new line."""

        return base_prompt

    def extract_citations(self, citations_text: str) -> List[Dict[str, str]]:
        """
        Extract citations from text formatted as "Citation Name | Citation URL"
        Returns a list of dictionaries in the required format
        """
        citations = []
        for line in citations_text.strip().split('\n'):
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    citation_name = parts[0].strip()
                    citation_url = parts[1].strip()
                    citations.append({
                        "citations_name": citation_name,
                        "citation_url": citation_url
                    })
        return citations

    def parse_response(self, response: str) -> Dict[str, Any]:
        # Initialize with default values
        sections = {
            "title": "Untitled",
            "tax_research": "No research provided",
            "tax_citations": "No citations provided",
            "draft_client_response": "No draft provided",
            "clarifying_questions": "No questions provided",
            "confirmation": "No confirmation provided"
        }
        
        # Map section headers to JSON keys
        header_to_key = {
            "[TITLE]": "title",
            "[TAX_RESEARCH]": "tax_research",
            "[TAX_CITATIONS]": "tax_citations",
            "[DRAFT_CLIENT_RESPONSE]": "draft_client_response",
            "[CLARIFYING_QUESTIONS]": "clarifying_questions",
            "[CONFIRMATION]": "confirmation"
        }
        
        # Split the response into sections
        current_section = None
        current_content = []
        
        # Process each line
        for line in response.split('\n'):
            line_stripped = line.strip()
            # Check if this line is a section header
            if line_stripped in header_to_key:
                # Save the previous section if it exists
                if current_section and current_section in header_to_key:
                    sections[header_to_key[current_section]] = '\n'.join(current_content).strip()
                # Start new section
                current_section = line_stripped
                current_content = []
            # If we're in a section and the line isn't empty, add it to current content
            elif current_section and line.strip():
                current_content.append(line)
        
        # Save the last section
        if current_section and current_section in header_to_key and current_content:
            sections[header_to_key[current_section]] = '\n'.join(current_content).strip()
        
        # Extract structured citations but don't add to the sections dict yet
        # We'll handle it differently in the answer_question method
        extracted_citations = self.extract_citations(sections["tax_citations"])
        
        return sections, extracted_citations

    def answer_question(self, query: str, prompt_params: Dict[str, str] = None) -> Dict[str, Any]:
        # Set default prompt params if none provided
        if prompt_params is None:
            prompt_params = {}
        
        # Retrieve context
        context = self.query_engine.retrieve_context(query)
        
        # Generate response
        prompt = self.generate_response_prompt(query, context, prompt_params)
        response = self.llm.invoke(prompt)
        
        # Parse response
        try:
            sections, citations = self.parse_response(response.content)
            
            # Create the response structure with citations as a separate field
            result = sections.copy()
            result["citations"] = citations
            
            return result
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw response: {response.content}")
            raise

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
async def query_tax_law(request: Request):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    try:
        # Parse request body
        try:
            body = await request.body()
            data = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        
        # Log the received data for debugging
        print(f"Received data: {data}")
        
        # Get the query from the request
        query_text = data.get("query")
        if not query_text:
            # FlutterFlow might send the query as 'message'
            query_text = data.get("message")
            if not query_text:
                raise HTTPException(status_code=400, detail="No query or message found in request")
        
        # Extract prompt parameters - ensure we have a default for promptCitations
        prompt_params = {
            'promptTitle': data.get('promptTitle'),
            'promptResearch': data.get('promptResearch'),
            'promptCitations': data.get('promptCitations', "Format each citation exactly as:\n\"Citation Name | Citation URL\" (one per line)\nExample: \"ITAA 1997 26-47 | https://www.ato.gov.au/law/view/document?docid=PAC/19970038/26-47\""),
            'promptClientResponse': data.get('promptClientResponse'), 
            'promptClarifyingQuestions': data.get('promptClarifyingQuestions'),
            'promptConfirmation': data.get('promptConfirmation')
        }
        
        # Remove None values from prompt_params
        prompt_params = {k: v for k, v in prompt_params.items() if v is not None}
        
        # Process the query
        result = rag_instance.answer_question(query_text, prompt_params)
        
        # Ensure the result has the correct format for citations
        if "citations" not in result or not isinstance(result["citations"], list):
            result["citations"] = []
        
        # Return as JSONResponse to ensure proper JSON formatting
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
