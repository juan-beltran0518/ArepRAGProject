from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_EMBEDDINGS_MODEL: str = "text-embedding-3-large"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-index"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"

settings = Settings()
