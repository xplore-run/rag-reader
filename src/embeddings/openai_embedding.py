"""OpenAI embeddings implementation."""

from typing import List, Union, Optional
import numpy as np
import openai
from openai import OpenAI
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

from .base import EmbeddingModel

logger = logging.getLogger(__name__)


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embeddings implementation."""
    
    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    # Max tokens per model
    MODEL_MAX_TOKENS = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
    }
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None
    ):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (if not set in environment)
            dimensions: Optional dimensions for models that support it
        """
        self.model_name = model_name or "text-embedding-3-small"
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
        
        # Set dimensions
        if dimensions:
            self._dimension = dimensions
        else:
            self._dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)
        
        self._max_seq_length = self.MODEL_MAX_TOKENS.get(model_name, 8191)
        
        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(
            f"Initialized OpenAI embedding model: {model_name}, "
            f"dimension: {self._dimension}"
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text."""
        if isinstance(text, str):
            text = [text]
        
        # Check token limits
        for t in text:
            token_count = len(self.encoding.encode(t))
            if token_count > self._max_seq_length:
                logger.warning(
                    f"Text exceeds token limit ({token_count} > {self._max_seq_length}). "
                    "It will be truncated."
                )
        
        # Call OpenAI API
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=self._dimension if self.model_name.startswith("text-embedding-3") else None
        )
        
        # Extract embeddings
        embeddings = np.array([item.embedding for item in response.data])
        
        # Return single embedding if input was single string
        if len(embeddings) == 1:
            return embeddings[0]
        
        return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Note: OpenAI allows up to 2048 embedding inputs per request.
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_text(batch)
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        return np.vstack(all_embeddings)
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self._dimension
    
    @property
    def max_seq_length(self) -> int:
        """Get the maximum sequence length in tokens."""
        return self._max_seq_length
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (uses model limit if not specified)
            
        Returns:
            Truncated text
        """
        max_tokens = max_tokens or self._max_seq_length
        
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def estimate_cost(self, num_tokens: int) -> float:
        """
        Estimate cost for embedding tokens.
        
        Args:
            num_tokens: Number of tokens to embed
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10,
        }
        
        price_per_million = pricing.get(self.model_name, 0.10)
        return (num_tokens / 1_000_000) * price_per_million