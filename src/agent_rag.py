from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pinecone import Pinecone
from src.settings import settings

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    AgentExecutor = None
    create_tool_calling_agent = None

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

vector_store = get_vector_store()

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def create_rag_agent():
    if AgentExecutor is None or create_tool_calling_agent is None:
        return None
        
    model = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
        base_url="http://localhost:11434"
    )
    
    tools = [retrieve_context]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You have access to a tool that retrieves context from a blog post. Use the tool to help answer user queries."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    try:
        agent = create_tool_calling_agent(model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return agent_executor
    except Exception:
        return None

def simple_agent_query(query: str) -> str:
    retrieved_docs = vector_store.similarity_search(query, k=2)
    
    context = "\n\n".join([
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    ])
    
    model = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
        base_url="http://localhost:11434"
    )
    
    prompt = f"""You have access to context from a blog post about LLM-powered autonomous agents.

Context:
{context}

User query: {query}

Please provide a helpful answer based on the context above."""
    
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content

def run_agent_query(query: str) -> str:
    agent = create_rag_agent()
    
    if agent is None:
        return simple_agent_query(query)
    
    try:
        result = agent.invoke({"input": query})
        return result["output"]
    except Exception:
        return simple_agent_query(query)

if __name__ == "__main__":
    queries = [
        "What is task decomposition?",
        "What are common ways of doing task decomposition?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        answer = run_agent_query(query)
        print(f"Answer: {answer}\n")
