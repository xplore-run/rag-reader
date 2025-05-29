"""Text chunking and preprocessing module for RAG pipeline."""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import tiktoken
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    document_id: str
    document_name: str
    page_numbers: List[int]
    start_char: int
    end_char: int
    metadata: Dict[str, any]


class TextChunker:
    """Handles text chunking with various strategies."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        method: str = "recursive",
        separator: str = "\n\n"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of overlapping characters between chunks
            method: Chunking method - 'recursive', 'character', 'token', 'sentence'
            separator: Primary separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method
        self.separator = separator
        
        # Initialize the appropriate splitter
        self.splitter = self._create_splitter()
    
    def _create_splitter(self):
        """Create the appropriate text splitter based on method."""
        if self.method == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        elif self.method == "character":
            return CharacterTextSplitter(
                separator=self.separator,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.method == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.method == "sentence":
            return SentenceTransformersTokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown chunking method: {self.method}")
    
    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page_mapping: Optional[Dict[int, str]] = None
    ) -> List[TextChunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            text: Full document text
            document_id: Unique document identifier
            document_name: Document file name
            page_mapping: Optional mapping of character positions to page numbers
            
        Returns:
            List of TextChunk objects
        """
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Split text into chunks
        chunks = self.splitter.split_text(text)
        
        # Create TextChunk objects
        text_chunks = []
        current_pos = 0
        
        for idx, chunk_text in enumerate(chunks):
            # Find chunk position in original text
            start_pos = text.find(chunk_text, current_pos)
            end_pos = start_pos + len(chunk_text)
            
            # Determine page numbers for this chunk
            page_numbers = []
            if page_mapping:
                page_numbers = self._get_page_numbers(
                    start_pos, end_pos, page_mapping
                )
            
            # Create chunk object
            chunk = TextChunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{idx}",
                document_id=document_id,
                document_name=document_name,
                page_numbers=page_numbers,
                start_char=start_pos,
                end_char=end_pos,
                metadata={
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "method": self.method
                }
            )
            
            text_chunks.append(chunk)
            current_pos = start_pos + 1
        
        return text_chunks
    
    def chunk_by_sections(
        self,
        text: str,
        document_id: str,
        document_name: str,
        section_headers: Optional[List[str]] = None
    ) -> List[TextChunk]:
        """
        Chunk document by sections (useful for technical docs).
        
        Args:
            text: Full document text
            document_id: Unique document identifier
            document_name: Document file name
            section_headers: List of section header patterns
            
        Returns:
            List of TextChunk objects
        """
        if not section_headers:
            # Default patterns for technical documentation
            section_headers = [
                r"^#{1,6}\s+.*$",  # Markdown headers
                r"^\d+\.?\s+[A-Z].*$",  # Numbered sections
                r"^[A-Z][A-Z\s]{2,}$",  # All caps headers
                r"^(Chapter|Section|Part)\s+\d+",  # Chapter/Section markers
            ]
        
        # Find all section boundaries
        sections = []
        current_section = {"start": 0, "text": ""}
        
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            # Check if line matches any header pattern
            is_header = any(
                re.match(pattern, line.strip(), re.MULTILINE) 
                for pattern in section_headers
            )
            
            if is_header and current_section["text"]:
                # Save current section
                sections.append(current_section)
                current_section = {
                    "start": current_pos,
                    "text": line + '\n',
                    "header": line.strip()
                }
            else:
                current_section["text"] += line + '\n'
            
            current_pos += len(line) + 1
        
        # Don't forget the last section
        if current_section["text"]:
            sections.append(current_section)
        
        # Now chunk each section if it's too large
        text_chunks = []
        for idx, section in enumerate(sections):
            section_text = section["text"]
            
            if len(section_text) <= self.chunk_size:
                # Section is small enough to be a single chunk
                chunk = TextChunk(
                    text=section_text,
                    chunk_id=f"{document_id}_section_{idx}",
                    document_id=document_id,
                    document_name=document_name,
                    page_numbers=[],
                    start_char=section["start"],
                    end_char=section["start"] + len(section_text),
                    metadata={
                        "section_index": idx,
                        "section_header": section.get("header", ""),
                        "is_section_chunk": True
                    }
                )
                text_chunks.append(chunk)
            else:
                # Section needs to be split further
                sub_chunks = self.chunk_document(
                    section_text,
                    document_id,
                    document_name
                )
                # Update chunk IDs and metadata
                for j, chunk in enumerate(sub_chunks):
                    chunk.chunk_id = f"{document_id}_section_{idx}_chunk_{j}"
                    chunk.metadata.update({
                        "section_index": idx,
                        "section_header": section.get("header", ""),
                        "is_section_chunk": False
                    })
                text_chunks.extend(sub_chunks)
        
        return text_chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        
        # Normalize quotes and apostrophes
        text = re.sub(r'[''`]', "'", text)
        text = re.sub(r'[""]', '"', text)
        
        return text.strip()
    
    def _get_page_numbers(
        self,
        start_pos: int,
        end_pos: int,
        page_mapping: Dict[int, str]
    ) -> List[int]:
        """
        Determine which pages a chunk spans.
        
        Args:
            start_pos: Start character position
            end_pos: End character position
            page_mapping: Mapping of character positions to page numbers
            
        Returns:
            List of page numbers
        """
        pages = set()
        for pos in range(start_pos, end_pos):
            if pos in page_mapping:
                pages.add(page_mapping[pos])
        return sorted(list(pages))
    
    def calculate_token_count(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Calculate token count for a text using tiktoken.
        
        Args:
            text: Text to count tokens for
            model: Model name for tokenizer
            
        Returns:
            Number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    def chunk_for_token_limit(
        self,
        text: str,
        max_tokens: int,
        model: str = "gpt-3.5-turbo"
    ) -> List[str]:
        """
        Chunk text based on token limits.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            model: Model name for tokenizer
            
        Returns:
            List of text chunks
        """
        # Use token-based splitter
        splitter = TokenTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=int(max_tokens * 0.1),
            model_name=model
        )
        
        return splitter.split_text(text)


def main():
    """Test text chunking."""
    sample_text = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly 
    programmed.
    
    ## Types of Machine Learning
    
    There are three main types of machine learning:
    
    1. Supervised Learning
    2. Unsupervised Learning
    3. Reinforcement Learning
    
    ### Supervised Learning
    
    In supervised learning, the algorithm learns from labeled training data.
    """
    
    # Test different chunking methods
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    chunks = chunker.chunk_document(
        sample_text,
        document_id="test_doc",
        document_name="test.pdf"
    )
    
    print(f"Created {len(chunks)} chunks")
    for chunk in chunks:
        print(f"\nChunk {chunk.chunk_id}:")
        print(f"Text: {chunk.text[:50]}...")
        print(f"Length: {len(chunk.text)}")


if __name__ == "__main__":
    main()