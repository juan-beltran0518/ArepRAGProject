"""
Consultar usando Ollama (100% gratuito y local)
Ejecutar como: python -m src.query_ollama
"""
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from src.settings import settings
from src.prompts import SYSTEM_PROMPT, USER_PROMPT

def build_retriever():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    # Usa el índice de Ollama
    index = pc.Index("rag-index-ollama")
    vs = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="ollama",
    )
    return vs.as_retriever(search_kwargs={"k": 4})

def rag_answer(question: str) -> str:
    retriever = build_retriever()
    # Usa llama3.2:1b o cualquier modelo que tengas en Ollama
    llm = OllamaLLM(model="llama3.2:1b", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
         "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain.invoke(question)

if __name__ == "__main__":
    print("🦙 Usando Ollama (modelos locales gratuitos)")
    print("Modelo de chat: llama3.2")
    print("Modelo de embeddings: nomic-embed-text\n")
    pregunta = input("Pregunta: ")
    respuesta = rag_answer(pregunta)
    print("\nRespuesta:\n", respuesta)
