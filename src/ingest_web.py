import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.settings import settings

def ensure_index(pc: Pinecone, name: str, dim: int, cloud: str, region: str):
    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

def main():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    index_name = "rag-index-ollama"
    dim = 768
    ensure_index(pc, index_name, dim, settings.PINECONE_CLOUD, settings.PINECONE_REGION)
    
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="web-blog",
    )
    
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"Indexed {len(document_ids)} documents")

if __name__ == "__main__":
    main()
