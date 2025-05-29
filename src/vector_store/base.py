"""Base interface for vector store implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class Document:
    """Represents a document with embeddings."""
    id: str
    text: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Represents a search result."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Retrieve documents by IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the store."""
        pass