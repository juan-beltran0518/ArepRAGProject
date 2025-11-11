from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from src.settings import settings
from src.prompts import SYSTEM_PROMPT, USER_PROMPT

app = FastAPI(
    title="RAG API con Ollama",
    description="API de preguntas y respuestas usando modelos locales gratuitos (Ollama)",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4
    model: Optional[str] = "llama3.2:1b"

class QuestionResponse(BaseModel):
    answer: str
    model_used: str
    embedding_model: str
    sources_found: int

def build_retriever(top_k: int = 4):
    """Construye retriever con embeddings de Ollama"""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    index = pc.Index("rag-index-ollama")
    vs = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="ollama",
    )
    return vs.as_retriever(search_kwargs={"k": top_k})

def rag_answer(question: str, model: str = "llama3.2:1b", top_k: int = 4) -> tuple[str, int]:
    """Genera respuesta usando RAG con Ollama"""
    retriever = build_retriever(top_k=top_k)
    llm = OllamaLLM(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    # Recuperar documentos
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    
    # Generar respuesta
    chain = prompt | llm
    answer = chain.invoke({"question": question, "context": context})
    
    return answer, len(docs)

@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "message": "RAG API con Ollama (100% gratis y local)",
        "endpoints": {
            "/ask": "POST - Hacer una pregunta",
            "/health": "GET - Estado del servicio",
            "/models": "GET - Modelos disponibles"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Verifica el estado del servicio"""
    try:
        # Verificar que Ollama esté disponible
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        # Hacer una prueba simple
        embeddings.embed_query("test")
        return {
            "status": "healthy",
            "ollama": "running",
            "embedding_model": "nomic-embed-text",
            "index": "rag-index-ollama"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama no está disponible: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """Lista los modelos disponibles en Ollama"""
    return {
        "embedding_model": "nomic-embed-text",
        "chat_models": [
            {
                "name": "llama3.2:1b",
                "description": "Modelo ligero y rápido (recomendado)",
                "size": "~1.3 GB"
            },
            {
                "name": "llama3.2:3b",
                "description": "Balance entre velocidad y calidad",
                "size": "~2 GB"
            },
            {
                "name": "qwen2.5:3b",
                "description": "Alternativa ligera",
                "size": "~2 GB"
            }
        ],
        "note": "Descarga modelos con: ollama pull <nombre>"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint principal para hacer preguntas sobre los documentos indexados
    
    - **question**: La pregunta a responder
    - **top_k**: Número de documentos a recuperar (default: 4)
    - **model**: Modelo de Ollama a usar (default: llama3.2:1b)
    """
    try:
        answer, sources_count = rag_answer(
            question=request.question,
            model=request.model,
            top_k=request.top_k
        )
        
        return QuestionResponse(
            answer=answer,
            model_used=request.model,
            embedding_model="nomic-embed-text",
            sources_found=sources_count
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la pregunta: {str(e)}"
        )

@app.post("/ingest")
async def trigger_ingest():
    """
    Endpoint para iniciar la ingesta de documentos
    (Ejecuta src.ingest_ollama en segundo plano)
    """
    import threading
    from src import ingest_ollama
    
    def run_ingest():
        ingest_ollama.main()
    
    thread = threading.Thread(target=run_ingest)
    thread.start()
    
    return {
        "status": "started",
        "message": "Ingesta iniciada en segundo plano",
        "note": "Los documentos en data/ serán procesados"
    }

if __name__ == "__main__":
    import uvicorn
    print("🦙 Iniciando API RAG con Ollama (100% gratis)")
    print("📍 Documentación interactiva: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
