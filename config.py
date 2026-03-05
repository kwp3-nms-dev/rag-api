from pydantic_settings import BaseSettings


ALLOWED_COLLECTIONS = {
    "cybersecurity_kb",
    "accounting_software_kb",
    "it_infrastructure_kb",
    "sop_docs",
    "microsoft_admin_kb",
}


class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    OLLAMA_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "qwen3-embedding:4b"
    LLM_MODEL: str = "qwen3.5:9b"
    TOP_K: int = 5
    API_KEY: str  # Required — no default, must be set in .env

    class Config:
        env_file = ".env"


settings = Settings()
