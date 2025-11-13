from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
from src.agent_rag import run_agent_query
from src.chain_rag import run_chain_query

app = FastAPI(title="RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    method: Literal["agent", "chain"] = "chain"
    include_sources: Optional[bool] = False

class QueryResponse(BaseModel):
    answer: str
    method: str
    sources_count: int
    sources: Optional[list] = None

@app.get("/")
async def root():
    return {
        "endpoints": {
            "/query": "POST",
            "/health": "GET"
        }
    }

@app.get("/health")
async def health():
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        embeddings.embed_query("test")
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        if request.method == "agent":
            answer = run_agent_query(request.question)
            sources = []
            sources_count = 0
        else:
            answer, docs = run_chain_query(request.question)
            sources_count = len(docs)
            sources = []
            if request.include_sources:
                sources = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ]
        
        return QueryResponse(
            answer=answer,
            method=request.method,
            sources_count=sources_count,
            sources=sources if request.include_sources else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_web():
    import threading
    from src import ingest_web
    
    def run_ingest():
        ingest_web.main()
    
    thread = threading.Thread(target=run_ingest)
    thread.start()
    return {"status": "started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
