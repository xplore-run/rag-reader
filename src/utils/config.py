"""Configuration management for the chatbot."""

from typing import Dict, Any, Optional
import os
import yaml
import json
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    type: str = Field(default="sentence_transformer", description="Embedding model type")
    model_name: Optional[str] = Field(default=None, description="Model name")
    device: Optional[str] = Field(default=None, description="Device to use")
    dimensions: Optional[int] = Field(default=None, description="Embedding dimensions")


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    type: str = Field(default="chromadb", description="Vector store type")
    collection_name: str = Field(default="tech_docs", description="Collection name")
    persist_directory: Optional[str] = Field(default="./chroma_db", description="Persistence directory")


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default="openai", description="LLM provider")
    model: Optional[str] = Field(default="gpt-4o-mini", description="Model name")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=1000, description="Max tokens to generate")


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    method: str = Field(default="recursive", description="Chunking method")
    chunk_size: int = Field(default=1000, description="Chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")


class RAGConfig(BaseModel):
    """RAG configuration."""
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0.0, description="Minimum relevance score")
    rerank: bool = Field(default=True, description="Whether to rerank results")
    include_neighbors: bool = Field(default=True, description="Include neighboring chunks")
    prompt_template: str = Field(default="default", description="Prompt template name")


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    
    # Component configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    
    # Application settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default="logs/chatbot.log", env="LOG_FILE")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from env vars
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Load settings from YAML file."""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        # Create settings - this will also load from .env file automatically
        # due to the Config class settings
        settings = cls(**config_data)
        return settings
    
    @classmethod
    def from_json(cls, json_path: str) -> "Settings":
        """Load settings from JSON file."""
        with open(json_path, 'r') as f:
            config_data = json.load(f) or {}
        
        # Create settings from JSON data only (env vars are loaded automatically by pydantic)
        settings = cls(**config_data)
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump(exclude_none=True)
    
    def save_yaml(self, path: str) -> None:
        """Save settings to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str) -> None:
        """Save settings to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Get application settings.
    
    Args:
        config_path: Optional path to config file (YAML or JSON)
        
    Returns:
        Settings instance
    """
    if config_path:
        path = Path(config_path)
        if path.suffix == '.yaml' or path.suffix == '.yml':
            return Settings.from_yaml(config_path)
        elif path.suffix == '.json':
            return Settings.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    # Load from environment and defaults
    return Settings()


# Create example configuration files
def create_example_configs():
    """Create example configuration files."""
    # Development config
    dev_config = {
        "embedding": {
            "type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2"
        },
        "vector_store": {
            "type": "chromadb",
            "persist_directory": "./chroma_db_dev"
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7
        },
        "rag": {
            "top_k": 5,
            "rerank": True
        },
        "log_level": "DEBUG"
    }
    
    # Production config
    prod_config = {
        "embedding": {
            "type": "openai",
            "model_name": "text-embedding-3-small"
        },
        "vector_store": {
            "type": "pinecone",
            "collection_name": "tech-docs-prod"
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.5
        },
        "rag": {
            "top_k": 10,
            "rerank": True,
            "score_threshold": 0.7
        },
        "log_level": "INFO"
    }
    
    # Save example configs
    os.makedirs("config", exist_ok=True)
    
    with open("config/dev.yaml", 'w') as f:
        yaml.dump(dev_config, f, default_flow_style=False)
    
    with open("config/prod.yaml", 'w') as f:
        yaml.dump(prod_config, f, default_flow_style=False)


if __name__ == "__main__":
    # Create example configs when run directly
    create_example_configs()
    print("Created example configuration files in config/")