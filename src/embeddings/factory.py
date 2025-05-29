"""Factory for creating embedding model instances."""

from typing import Optional, Dict, Any
import os
from .base import EmbeddingModel
from .sentence_transformer import SentenceTransformerEmbedding
from .openai_embedding import OpenAIEmbedding


class EmbeddingFactory:
    """Factory for creating embedding model instances."""
    
    @staticmethod
    def create(
        model_type: str = "sentence_transformer",
        **kwargs
    ) -> EmbeddingModel:
        """
        Create an embedding model instance.
        
        Args:
            model_type: Type of embedding model ('sentence_transformer', 'openai')
            **kwargs: Additional arguments for the specific model
            
        Returns:
            EmbeddingModel instance
        """
        model_type = model_type.lower()
        
        if model_type == "sentence_transformer" or model_type == "st":
            return EmbeddingFactory._create_sentence_transformer(**kwargs)
        elif model_type == "openai":
            return EmbeddingFactory._create_openai(**kwargs)
        else:
            raise ValueError(f"Unknown embedding model type: {model_type}")
    
    @staticmethod
    def _create_sentence_transformer(**kwargs) -> SentenceTransformerEmbedding:
        """Create Sentence Transformer model with default settings."""
        model_name = kwargs.get(
            'model_name',
            os.getenv('ST_MODEL_NAME', 'all-MiniLM-L6-v2')
        )
        device = kwargs.get('device', None)
        cache_folder = kwargs.get('cache_folder', './models/sentence_transformers')
        
        return SentenceTransformerEmbedding(
            model_name=model_name,
            device=device,
            cache_folder=cache_folder
        )
    
    @staticmethod
    def _create_openai(**kwargs) -> OpenAIEmbedding:
        """Create OpenAI embedding model with default settings."""
        model_name = kwargs.get(
            'model_name',
            os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        )
        api_key = kwargs.get('api_key', os.getenv('OPENAI_API_KEY'))
        dimensions = kwargs.get('dimensions', None)
        
        return OpenAIEmbedding(
            model_name=model_name,
            api_key=api_key,
            dimensions=dimensions
        )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> EmbeddingModel:
        """
        Create embedding model from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and 'config' keys
            
        Returns:
            EmbeddingModel instance
        """
        model_type = config.get('type', 'sentence_transformer')
        model_config = config.get('config', {})
        
        return EmbeddingFactory.create(model_type, **model_config)
    
    @staticmethod
    def get_recommended_model(use_case: str = "general") -> Dict[str, Any]:
        """
        Get recommended model configuration for a use case.
        
        Args:
            use_case: Use case ('general', 'qa', 'search', 'multilingual', 'fast')
            
        Returns:
            Configuration dictionary
        """
        recommendations = {
            "general": {
                "type": "sentence_transformer",
                "config": {
                    "model_name": "all-mpnet-base-v2"
                }
            },
            "qa": {
                "type": "sentence_transformer",
                "config": {
                    "model_name": "multi-qa-mpnet-base-dot-v1"
                }
            },
            "search": {
                "type": "sentence_transformer",
                "config": {
                    "model_name": "msmarco-distilbert-base-tas-b"
                }
            },
            "multilingual": {
                "type": "sentence_transformer",
                "config": {
                    "model_name": "paraphrase-multilingual-mpnet-base-v2"
                }
            },
            "fast": {
                "type": "sentence_transformer",
                "config": {
                    "model_name": "all-MiniLM-L6-v2"
                }
            },
            "high_quality": {
                "type": "openai",
                "config": {
                    "model_name": "text-embedding-3-small"
                }
            }
        }
        
        return recommendations.get(use_case, recommendations["general"])