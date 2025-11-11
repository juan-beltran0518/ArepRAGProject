"""
Consultar el índice Pinecone: embed la pregunta, recuperar top_k, y opcionalmente enviar al modelo de chat para generar respuesta.
Ejecutar como: python -m src.query
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from src.settings import settings
from src.prompts import SYSTEM_PROMPT, USER_PROMPT

def build_retriever():
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_EMBEDDINGS_MODEL,
    )
    vs = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace="default",
    )
    return vs.as_retriever(search_kwargs={"k": 4})

def rag_answer(question: str) -> str:
    retriever = build_retriever()
    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_CHAT_MODEL, temperature=0)

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
    return chain.invoke(question).content

if __name__ == "__main__":
    pregunta = input("Pregunta: ")
    respuesta = rag_answer(pregunta)
    print("\nRespuesta:\n", respuesta)
