from fastapi import FastAPI
from pydantic import BaseModel
from src.query import rag_answer

app = FastAPI(title="RAG API")

class Q(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Q):
    """Endpoint para hacer preguntas sobre los documentos indexados"""
    return {"answer": rag_answer(q.question)}

@app.get("/")
def root():
    return {"msg": "RAG API con LangChain, Pinecone y OpenAI"}
