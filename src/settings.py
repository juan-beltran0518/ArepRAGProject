from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-index-ollama"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"

settings = Settings()
