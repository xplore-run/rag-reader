"""RAG retrieval system for finding relevant documents."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from ..embeddings.base import EmbeddingModel
from ..vector_store.base import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with context."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    context: Optional[str] = None  # Additional context from neighboring chunks


class RAGRetriever:
    """Retrieves relevant documents for RAG pipeline."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        rerank: bool = True,
        include_neighbors: bool = True
    ):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_model: Embedding model instance
            rerank: Whether to rerank results
            include_neighbors: Whether to include neighboring chunks
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.rerank = rerank
        self.include_neighbors = include_neighbors
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            score_threshold: Minimum score threshold
            
        Returns:
            List of retrieval results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search vector store
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k * 2 if self.rerank else k,  # Get more results if reranking
            filter=filter
        )
        
        # Filter by score threshold
        # Note: Some vector stores return negative scores (e.g., cosine distance)
        # so we need to be careful with the threshold
        if score_threshold > 0:
            search_results = [r for r in search_results if r.score >= score_threshold]
        else:
            # If threshold is 0 or negative, don't filter by score
            # This handles cases where scores are negative (distances)
            pass
        
        # Convert to retrieval results
        retrieval_results = []
        for result in search_results:
            retrieval_result = RetrievalResult(
                chunk_id=result.id,
                text=result.text,
                score=result.score,
                metadata=result.metadata
            )
            retrieval_results.append(retrieval_result)
        
        # Rerank if enabled
        if self.rerank and len(retrieval_results) > k:
            retrieval_results = self._rerank_results(
                query, retrieval_results, k
            )
        
        # Include neighboring chunks if enabled
        if self.include_neighbors:
            retrieval_results = self._add_neighbor_context(retrieval_results)
        
        # Limit to k results
        retrieval_results = retrieval_results[:k]
        
        logger.info(f"Retrieved {len(retrieval_results)} results for query")
        return retrieval_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder or other methods.
        
        Args:
            query: Query text
            results: Initial retrieval results
            k: Number of results to keep
            
        Returns:
            Reranked results
        """
        # Simple reranking based on keyword overlap
        # In production, use a cross-encoder model
        
        query_tokens = set(query.lower().split())
        
        for result in results:
            result_tokens = set(result.text.lower().split())
            
            # Calculate keyword overlap
            overlap = len(query_tokens & result_tokens)
            overlap_ratio = overlap / len(query_tokens) if query_tokens else 0
            
            # Adjust score based on overlap
            result.score = result.score * (1 + overlap_ratio * 0.5)
        
        # Sort by adjusted score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:k]
    
    def _add_neighbor_context(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Add context from neighboring chunks.
        
        Args:
            results: Retrieval results
            
        Returns:
            Results with neighbor context
        """
        for result in results:
            # Extract chunk index from metadata
            chunk_index = result.metadata.get('chunk_index')
            document_id = result.metadata.get('document_id')
            
            if chunk_index is not None and document_id:
                # Try to get previous and next chunks
                prev_chunk_id = f"{document_id}_chunk_{chunk_index - 1}"
                next_chunk_id = f"{document_id}_chunk_{chunk_index + 1}"
                
                context_parts = []
                
                # Get previous chunk
                try:
                    prev_chunks = self.vector_store.get_by_ids([prev_chunk_id])
                    if prev_chunks:
                        context_parts.append(f"[Previous context]\n{prev_chunks[0].text}")
                except:
                    pass
                
                # Get next chunk
                try:
                    next_chunks = self.vector_store.get_by_ids([next_chunk_id])
                    if next_chunks:
                        context_parts.append(f"[Following context]\n{next_chunks[0].text}")
                except:
                    pass
                
                if context_parts:
                    result.context = "\n\n".join(context_parts)
        
        return results
    
    def retrieve_multi_query(
        self,
        queries: List[str],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        aggregation: str = "max"
    ) -> List[RetrievalResult]:
        """
        Retrieve using multiple query variations.
        
        Args:
            queries: List of query variations
            k: Number of results to return
            filter: Optional metadata filter
            aggregation: How to aggregate scores ('max', 'mean')
            
        Returns:
            Aggregated retrieval results
        """
        # Get results for each query
        all_results = {}
        
        for query in queries:
            results = self.retrieve(query, k=k*2, filter=filter)
            
            for result in results:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = {
                        'result': result,
                        'scores': []
                    }
                all_results[result.chunk_id]['scores'].append(result.score)
        
        # Aggregate scores
        final_results = []
        for chunk_id, data in all_results.items():
            if aggregation == "max":
                final_score = max(data['scores'])
            elif aggregation == "mean":
                final_score = np.mean(data['scores'])
            else:
                final_score = data['scores'][0]
            
            result = data['result']
            result.score = final_score
            final_results.append(result)
        
        # Sort by score and return top k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]
    
    def retrieve_with_feedback(
        self,
        query: str,
        k: int = 5,
        positive_examples: Optional[List[str]] = None,
        negative_examples: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve with relevance feedback.
        
        Args:
            query: Query text
            k: Number of results
            positive_examples: Positive example texts
            negative_examples: Negative example texts
            
        Returns:
            Retrieval results adjusted by feedback
        """
        # Get query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Adjust embedding based on feedback
        if positive_examples:
            pos_embeddings = self.embedding_model.embed_text(positive_examples)
            if pos_embeddings.ndim == 1:
                pos_embeddings = pos_embeddings.reshape(1, -1)
            query_embedding += 0.3 * np.mean(pos_embeddings, axis=0)
        
        if negative_examples:
            neg_embeddings = self.embedding_model.embed_text(negative_examples)
            if neg_embeddings.ndim == 1:
                neg_embeddings = neg_embeddings.reshape(1, -1)
            query_embedding -= 0.2 * np.mean(neg_embeddings, axis=0)
        
        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search with adjusted embedding
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k
        )
        
        # Convert to retrieval results
        retrieval_results = []
        for result in search_results:
            retrieval_result = RetrievalResult(
                chunk_id=result.id,
                text=result.text,
                score=result.score,
                metadata=result.metadata
            )
            retrieval_results.append(retrieval_result)
        
        return retrieval_results