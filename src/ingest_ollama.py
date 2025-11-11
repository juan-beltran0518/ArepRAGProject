import glob
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.settings import settings

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_documents():
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        docs.extend(PyPDFLoader(path).load())
    for path in glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "*.md")):
        docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs

def ensure_index(pc: Pinecone, name: str, dim: int, cloud: str, region: str):
    """Crea un índice 'dense' si no existe"""
    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"✓ Índice '{name}' creado")

def main():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    # Ollama embeddings (gratuito, local)
    # Dimensión: 768 para nomic-embed-text
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # Prepara índice con dimensión 768 para nomic-embed-text
    # Usa un índice diferente para Ollama
    index_name = "rag-index-ollama"
    dim = 768
    ensure_index(pc, index_name, dim, settings.PINECONE_CLOUD, settings.PINECONE_REGION)

    # Carga y separa documentos
    docs = load_documents()
    if not docs:
        print("No se encontraron documentos en data/")
        return
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Procesando {len(chunks)} chunks...")
    # Upsert a Pinecone
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="ollama",
    )
    vectorstore.add_documents(chunks)
    print(f"✓ Ingestión completada: {len(chunks)} chunks indexados en '{index_name}'")

if __name__ == "__main__":
    print("Usando Ollama para embeddings (gratuito)")
    print("Asegúrate de tener Ollama corriendo: ollama serve")
    print("Y el modelo descargado: ollama pull nomic-embed-text\n")
    main()
