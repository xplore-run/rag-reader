"""Factory for creating vector store instances."""

from typing import Optional, Dict, Any
import os
from .base import VectorStore
from .chromadb_store import ChromaDBStore


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create(
        store_type: str = "chromadb",
        **kwargs
    ) -> VectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: Type of vector store ('chromadb', 'pinecone', 'faiss')
            **kwargs: Additional arguments for the specific store
            
        Returns:
            VectorStore instance
        """
        store_type = store_type.lower()
        
        if store_type == "chromadb":
            return VectorStoreFactory._create_chromadb(**kwargs)
        elif store_type == "pinecone":
            return VectorStoreFactory._create_pinecone(**kwargs)
        elif store_type == "faiss":
            return VectorStoreFactory._create_faiss(**kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
    
    @staticmethod
    def _create_chromadb(**kwargs) -> ChromaDBStore:
        """Create ChromaDB store with default settings."""
        collection_name = kwargs.get(
            'collection_name',
            os.getenv('CHROMA_COLLECTION_NAME', 'tech_docs')
        )
        persist_directory = kwargs.get(
            'persist_directory',
            os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        )
        embedding_function = kwargs.get('embedding_function', None)
        
        return ChromaDBStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
    
    @staticmethod
    def _create_pinecone(**kwargs):
        """Create Pinecone store (to be implemented)."""
        # Import here to avoid dependency if not using Pinecone
        from .pinecone_store import PineconeStore
        
        api_key = kwargs.get('api_key', os.getenv('PINECONE_API_KEY'))
        environment = kwargs.get('environment', os.getenv('PINECONE_ENVIRONMENT'))
        index_name = kwargs.get('index_name', os.getenv('PINECONE_INDEX_NAME', 'tech-docs'))
        
        if not api_key or not environment:
            raise ValueError("Pinecone requires API key and environment")
        
        return PineconeStore(
            api_key=api_key,
            environment=environment,
            index_name=index_name
        )
    
    @staticmethod
    def _create_faiss(**kwargs):
        """Create FAISS store (to be implemented)."""
        # Import here to avoid dependency if not using FAISS
        from .faiss_store import FAISSStore
        
        index_path = kwargs.get('index_path', './faiss_index')
        dimension = kwargs.get('dimension', 768)  # Default for sentence-transformers
        
        return FAISSStore(
            index_path=index_path,
            dimension=dimension
        )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> VectorStore:
        """
        Create vector store from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            VectorStore instance
        """
        store_type = config.get('type', 'chromadb')
        store_config = config.get('config', {})
        
        return VectorStoreFactory.create(store_type, **store_config)