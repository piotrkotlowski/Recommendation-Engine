from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    RAG_CHROMA_COLLECTION: str = "products"
    RAG_CHROMA_METRIC: str = "cosine"
    RAG_HNSW_M: int = 32
    RAG_HNSW_EF_CONSTRUCTION: int = 400
    RAG_HNSW_EF_SEARCH: int = 128
    RAG_EMB_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_PRODUCT_DATA_PATH: str = "/data_ingest/ProductsRAG.json"
    RAG_CHROMA_BATCH_SIZE: int = 32
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CHROMA_HOST: str ="chroma_db"
    CHROMA_PORT: int =8000

    ALS_PARAM_PATH : str ="als_trainer/params/als_params.json"

    OLLAMA_MODEL:str ="mistral:7b"
    OLLAMA_URL: str = 'http://ollama:11434'
    PG_HOST: str ="postgres_db"
    

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    PG_PORT: int = 5432
    PG_USER: str = "postgres"
    PG_PASSWORD: str = "mouse"
    PG_DATABASE: str = "amazon"

    PG_Ingestion_Csv_Users: str = "/data_ingest/users.csv"
    PG_Ingestion_Csv_Items: str = "/data_ingest/items.csv"
    PG_Ingestion_Csv_Interactions: str = "/data_ingest/interactions.csv"


    @property
    def PG_DSN(self) -> str:
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DATABASE}"



    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
