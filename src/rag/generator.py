"""RAG generation component for creating answers from retrieved context."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .retriever import RetrievalResult
from ..llm.base import LLMClient, Message, Role

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Generated answer with metadata."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    tokens_used: int
    retrieval_results: List[RetrievalResult]


class PromptTemplate:
    """Manages prompt templates for RAG generation."""
    
    # Default RAG prompt template
    DEFAULT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the information provided in the context
- If the context doesn't contain enough information to answer the question, say so
- Be concise and direct
- Cite the source documents when possible

Answer:"""

    # Technical documentation prompt
    TECHNICAL_TEMPLATE = """You are a technical documentation expert. Answer the following question using ONLY the provided documentation.

Documentation Context:
{context}

Technical Question: {question}

Guidelines:
- Provide accurate technical information from the documentation
- Include code examples if present in the context
- Reference specific sections or page numbers when available
- If the documentation doesn't cover this topic, clearly state that
- Use technical terminology appropriately

Technical Answer:"""

    # Q&A prompt with citations
    QA_WITH_CITATIONS = """Based on the following documents, answer the question and provide citations.

Documents:
{context}

Question: {question}

Format your answer as:
1. Direct answer to the question
2. Supporting details from the documents
3. Citations in format [Source: document_name, page_number]

Answer with Citations:"""

    @staticmethod
    def get_template(template_name: str = "default") -> str:
        """Get a prompt template by name."""
        templates = {
            "default": PromptTemplate.DEFAULT_TEMPLATE,
            "technical": PromptTemplate.TECHNICAL_TEMPLATE,
            "qa_citations": PromptTemplate.QA_WITH_CITATIONS
        }
        return templates.get(template_name, PromptTemplate.DEFAULT_TEMPLATE)


class RAGGenerator:
    """Generates answers using retrieved context and LLM."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str = "default",
        max_context_length: int = 3000,
        include_sources: bool = True
    ):
        """
        Initialize RAG generator.
        
        Args:
            llm_client: LLM client for generation
            prompt_template: Name of prompt template to use
            max_context_length: Maximum context length in characters
            include_sources: Whether to include source information
        """
        self.llm_client = llm_client
        self.prompt_template = PromptTemplate.get_template(prompt_template)
        self.max_context_length = max_context_length
        self.include_sources = include_sources
    
    def generate(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        additional_context: Optional[str] = None
    ) -> GeneratedAnswer:
        """
        Generate answer from question and retrieved context.
        
        Args:
            question: User question
            retrieval_results: Retrieved document chunks
            additional_context: Optional additional context
            
        Returns:
            Generated answer with metadata
        """
        # Prepare context from retrieval results
        context = self._prepare_context(retrieval_results, additional_context)
        
        # Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # Generate answer (placeholder - will be implemented with actual LLM)
        # For now, return a structured response
        answer = self._generate_with_llm(prompt)
        
        # Extract sources
        sources = self._extract_sources(retrieval_results)
        
        # Calculate confidence based on retrieval scores
        avg_score = sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
        confidence = min(avg_score, 1.0)
        
        return GeneratedAnswer(
            answer=answer,
            sources=sources,
            confidence=confidence,
            tokens_used=len(prompt.split()),  # Placeholder
            retrieval_results=retrieval_results
        )
    
    def _prepare_context(
        self,
        retrieval_results: List[RetrievalResult],
        additional_context: Optional[str] = None
    ) -> str:
        """
        Prepare context from retrieval results.
        
        Args:
            retrieval_results: Retrieved chunks
            additional_context: Optional additional context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add additional context if provided
        if additional_context:
            context_parts.append(f"Background Information:\n{additional_context}\n")
        
        # Add retrieved chunks
        for i, result in enumerate(retrieval_results, 1):
            # Format chunk with metadata
            chunk_text = f"[Document {i}]"
            
            if self.include_sources:
                # Add source information
                doc_name = result.metadata.get('document_name', 'Unknown')
                page_nums = result.metadata.get('page_numbers', [])
                
                chunk_text += f" (Source: {doc_name}"
                if page_nums:
                    chunk_text += f", Pages: {', '.join(map(str, page_nums))}"
                chunk_text += ")"
            
            chunk_text += f"\n{result.text}"
            
            # Add neighbor context if available
            if result.context:
                chunk_text += f"\n\n{result.context}"
            
            context_parts.append(chunk_text)
        
        # Join all parts
        full_context = "\n\n---\n\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > self.max_context_length:
            full_context = full_context[:self.max_context_length] + "..."
            logger.warning("Context truncated due to length limit")
        
        return full_context
    
    def _generate_with_llm(self, prompt: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated answer
        """
        try:
            # Use the LLM client to generate response
            response = self.llm_client.complete(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error while generating the response. Please try again."
    
    def _extract_sources(self, retrieval_results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieval results.
        
        Args:
            retrieval_results: Retrieved chunks
            
        Returns:
            List of source information
        """
        sources = []
        seen_docs = set()
        
        for result in retrieval_results:
            doc_name = result.metadata.get('document_name', 'Unknown')
            
            if doc_name not in seen_docs:
                source_info = {
                    'document': doc_name,
                    'chunks': [],
                    'relevance_score': result.score
                }
                
                # Collect all chunks from this document
                for r in retrieval_results:
                    if r.metadata.get('document_name') == doc_name:
                        chunk_info = {
                            'chunk_id': r.chunk_id,
                            'pages': r.metadata.get('page_numbers', []),
                            'score': r.score
                        }
                        source_info['chunks'].append(chunk_info)
                
                sources.append(source_info)
                seen_docs.add(doc_name)
        
        # Sort by relevance
        sources.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return sources
    
    def generate_with_feedback(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        feedback: Dict[str, Any]
    ) -> GeneratedAnswer:
        """
        Generate answer with user feedback incorporated.
        
        Args:
            question: User question
            retrieval_results: Retrieved chunks
            feedback: User feedback (e.g., preference for detail level)
            
        Returns:
            Generated answer
        """
        # Adjust generation based on feedback
        detail_level = feedback.get('detail_level', 'normal')
        
        if detail_level == 'brief':
            self.max_context_length = 1500
            template = "Provide a brief answer: {question}\nContext: {context}\nBrief Answer:"
        elif detail_level == 'detailed':
            self.max_context_length = 5000
            template = "Provide a comprehensive answer: {question}\nContext: {context}\nDetailed Answer:"
        else:
            template = self.prompt_template
        
        # Generate with adjusted parameters
        context = self._prepare_context(retrieval_results)
        prompt = template.format(context=context, question=question)
        answer = self._generate_with_llm(prompt)
        
        sources = self._extract_sources(retrieval_results)
        
        return GeneratedAnswer(
            answer=answer,
            sources=sources,
            confidence=0.8,
            tokens_used=len(prompt.split()),
            retrieval_results=retrieval_results
        )