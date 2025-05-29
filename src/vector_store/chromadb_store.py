"""ChromaDB vector store implementation."""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging

from .base import VectorStore, Document, SearchResult

logger = logging.getLogger(__name__)


class ChromaDBStore(VectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(
        self,
        collection_name: str = "tech_docs",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize ChromaDB store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedding_function: Optional embedding function
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        else:
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        logger.info(f"Initialized ChromaDB store with collection: {collection_name}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to ChromaDB."""
        if not documents:
            return []
        
        ids = []
        texts = []
        embeddings = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc.id)
            texts.append(doc.text)
            if doc.embedding is not None:
                embeddings.append(doc.embedding.tolist())
            metadatas.append(doc.metadata)
        
        # Add to collection
        if embeddings:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        else:
            # Let ChromaDB generate embeddings
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        # Convert numpy array to list
        query_embedding_list = query_embedding.tolist()
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k,
            where=filter if filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert results to SearchResult objects
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = SearchResult(
                    id=results['ids'][0][i],
                    text=results['documents'][0][i],
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                search_results.append(result)
        
        return search_results
    
    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search using text query (ChromaDB will embed it)."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
            where=filter if filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert results to SearchResult objects
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = SearchResult(
                    id=results['ids'][0][i],
                    text=results['documents'][0][i],
                    score=1.0 - results['distances'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                search_results.append(result)
        
        return search_results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by ID."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Retrieve documents by IDs."""
        results = self.collection.get(
            ids=ids,
            include=["documents", "embeddings", "metadatas"]
        )
        
        documents = []
        for i in range(len(results['ids'])):
            doc = Document(
                id=results['ids'][i],
                text=results['documents'][i] if results['documents'] else "",
                embedding=np.array(results['embeddings'][i]) if results['embeddings'] else None,
                metadata=results['metadatas'][i] if results['metadatas'] else {}
            )
            documents.append(doc)
        
        return documents
    
    def persist(self) -> None:
        """Persist the vector store."""
        if self.persist_directory:
            # ChromaDB with persistent client auto-persists
            logger.info("ChromaDB data persisted")
        else:
            logger.warning("No persist directory configured")
    
    def load(self) -> None:
        """Load the vector store."""
        # ChromaDB with persistent client auto-loads
        if self.persist_directory:
            logger.info("ChromaDB data loaded")
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        # Get all document IDs
        all_ids = self.collection.get()['ids']
        if all_ids:
            self.collection.delete(ids=all_ids)
        logger.info("Cleared all documents from ChromaDB")
    
    def count(self) -> int:
        """Get the number of documents in the store."""
        return self.collection.count()
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """Update document metadata."""
        try:
            self.collection.update(
                ids=[id],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")