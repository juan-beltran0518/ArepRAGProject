
import glob
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
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
    """Crea un índice 'dense' si no existe (dim debe coincidir con tu modelo de embeddings)"""
    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

def main():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    # text-embedding-3-large produce 3072 dimensiones por defecto; ajusta si cambias de modelo
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_EMBEDDINGS_MODEL,
    )

    # Prepara índice
    dim = 3072  # para text-embedding-3-large
    ensure_index(pc, settings.PINECONE_INDEX_NAME, dim, settings.PINECONE_CLOUD, settings.PINECONE_REGION)

    # Carga y separa documentos
    docs = load_documents()
    if not docs:
        print("No se encontraron documentos en data/")
        return
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Procesando {len(chunks)} chunks...")
    # Upsert a Pinecone vía LangChain - usa el objeto Pinecone ya inicializado
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="default",
    )
    vectorstore.add_documents(chunks)
    print(f"✓ Ingestión completada: {len(chunks)} chunks indexados en '{settings.PINECONE_INDEX_NAME}'")

if __name__ == "__main__":
    main()
