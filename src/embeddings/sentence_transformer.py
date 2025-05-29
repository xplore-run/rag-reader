"""Sentence Transformers embedding implementation."""

from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm

from .base import EmbeddingModel

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence Transformers implementation for embeddings."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize Sentence Transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run on ('cpu', 'cuda', 'mps')
            cache_folder: Folder to cache models
        """
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.device = device
        
        # Load model
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=cache_folder
            )
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {self.model_name}: {e}")
            raise
        
        # Get model properties
        self._dimension = self.model.get_sentence_embedding_dimension()
        # Get max sequence length - try different attributes
        if hasattr(self.model, 'max_seq_length'):
            self._max_seq_length = self.model.max_seq_length
        elif hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'model_max_length'):
            self._max_seq_length = self.model.tokenizer.model_max_length
        else:
            self._max_seq_length = 512  # Default fallback
        
        logger.info(
            f"Loaded SentenceTransformer model: {self.model_name}, "
            f"dimension: {self._dimension}, "
            f"max_seq_length: {self._max_seq_length}"
        )
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text."""
        if isinstance(text, str):
            # Single text
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
        else:
            # List of texts
            embeddings = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(text) > 100
            )
            return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return np.array([])
        
        try:
            # Sentence transformers handles batching internally
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding batch of {len(texts)} texts: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self._dimension
    
    @property
    def max_seq_length(self) -> int:
        """Get the maximum sequence length."""
        return self._max_seq_length
    
    def encode_with_metadata(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[dict]:
        """
        Encode texts and return with metadata.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            List of dicts with text, embedding, and token count
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            for text, embedding in zip(batch_texts, batch_embeddings):
                # Approximate token count
                if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'tokenize'):
                    token_count = len(self.model.tokenizer.tokenize(text))
                else:
                    # Rough approximation: ~1.3 tokens per word
                    token_count = int(len(text.split()) * 1.3)
                
                results.append({
                    'text': text,
                    'embedding': embedding,
                    'token_count': token_count,
                    'truncated': token_count > self._max_seq_length
                })
        
        return results
    
    @staticmethod
    def list_available_models() -> List[str]:
        """List popular sentence transformer models."""
        return [
            "all-MiniLM-L6-v2",  # Fast, good quality
            "all-mpnet-base-v2",  # Best quality
            "multi-qa-mpnet-base-dot-v1",  # Optimized for Q&A
            "all-distilroberta-v1",  # Good balance
            "msmarco-distilbert-base-tas-b",  # Optimized for search
            "all-MiniLM-L12-v2",  # Larger, better quality
            "paraphrase-multilingual-mpnet-base-v2",  # Multilingual
        ]
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = np.dot(embeddings1, embeddings2.T)
        
        return similarity