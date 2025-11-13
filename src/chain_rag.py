from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone
from src.settings import settings

def get_vector_store():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    index = pc.Index("rag-index-ollama")
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="web-blog",
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    model = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
        base_url="http://localhost:11434"
    )
    
    template = """You are a helpful assistant. Use the following context to answer the query.

Context:
{context}

Query: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def run_chain_query(question: str) -> tuple[str, list]:
    rag_chain, retriever = create_rag_chain()
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return answer, docs

if __name__ == "__main__":
    question = "What is task decomposition?"
    answer, docs = run_chain_query(question)
    
    print(f"Question: {question}")
    print(f"Documents retrieved: {len(docs)}")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.metadata.get('source', 'N/A')}")
    print(f"\nAnswer: {answer}")
