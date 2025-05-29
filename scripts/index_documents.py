#!/usr/bin/env python3
"""Script to index PDF documents into the vector store."""

import sys
import os
from pathlib import Path
import argparse
from typing import List, Optional
import time
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pdf_processor.pdf_extractor import PDFExtractor
from src.pdf_processor.text_chunker import TextChunker
from src.embeddings.factory import EmbeddingFactory
from src.vector_store.factory import VectorStoreFactory
from src.vector_store.base import Document
from src.utils.config import get_settings
from src.utils.logger import setup_logging, logger


def index_documents(
    pdf_directory: str,
    config_path: Optional[str] = None,
    clear_existing: bool = False
) -> None:
    """
    Index PDF documents into vector store.
    
    Args:
        pdf_directory: Directory containing PDF files
        config_path: Optional path to configuration file
        clear_existing: Whether to clear existing data
    """
    # Load configuration
    settings = get_settings(config_path)
    
    # Set up logging
    setup_logging(
        log_level=settings.log_level,
        log_file=settings.log_file
    )
    
    logger.info("Starting document indexing process")
    logger.info(f"PDF directory: {pdf_directory}")
    
    # Initialize components
    logger.info("Initializing components...")
    
    # PDF extractor
    pdf_extractor = PDFExtractor(method="auto")
    
    # Text chunker
    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        method=settings.chunking.method
    )
    
    # Embedding model
    embedding_model = EmbeddingFactory.create(
        model_type=settings.embedding.type,
        model_name=settings.embedding.model_name,
        device=settings.embedding.device
    )
    
    # Vector store
    vector_store = VectorStoreFactory.create(
        store_type=settings.vector_store.type,
        collection_name=settings.vector_store.collection_name,
        persist_directory=settings.vector_store.persist_directory
    )
    
    # Clear existing data if requested
    if clear_existing:
        logger.warning("Clearing existing vector store data...")
        vector_store.clear()
    
    # Get PDF files
    pdf_path = Path(pdf_directory)
    pdf_files = list(pdf_path.glob("*.pdf"))
    
    # Also check subdirectories
    if not pdf_files:
        pdf_files = list(pdf_path.rglob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_directory}")
        logger.info("Make sure PDF files are in the train-data directory or its subdirectories")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    total_chunks = 0
    start_time = time.time()
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract text from PDF
            pdf_doc = pdf_extractor.extract_text(str(pdf_file))
            logger.info(f"Extracted {pdf_doc.total_pages} pages from {pdf_file.name}")
            
            # Chunk the text
            chunks = chunker.chunk_document(
                text=pdf_doc.text_content,
                document_id=pdf_file.stem,
                document_name=pdf_file.name,
                page_mapping=None  # TODO: Implement page mapping
            )
            logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
            
            # Create embeddings for chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = embedding_model.embed_batch(chunk_texts, batch_size=32)
            
            # Create Document objects
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = Document(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    metadata={
                        'document_id': chunk.document_id,
                        'document_name': chunk.document_name,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'page_numbers': ','.join(map(str, chunk.page_numbers)) if chunk.page_numbers else '',
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        **chunk.metadata
                    }
                )
                documents.append(doc)
            
            # Add to vector store
            vector_store.add_documents(documents)
            total_chunks += len(documents)
            
            logger.info(f"Successfully indexed {len(documents)} chunks from {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Persist vector store
    vector_store.persist()
    
    # Calculate stats
    elapsed_time = time.time() - start_time
    
    logger.info("=" * 50)
    logger.info("Indexing completed successfully!")
    logger.info(f"Total PDFs processed: {len(pdf_files)}")
    logger.info(f"Total chunks created: {total_chunks}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per PDF: {elapsed_time/len(pdf_files):.2f} seconds")
    logger.info("=" * 50)


def verify_index(config_path: Optional[str] = None) -> None:
    """
    Verify the indexed documents.
    
    Args:
        config_path: Optional path to configuration file
    """
    settings = get_settings(config_path)
    
    # Initialize vector store
    vector_store = VectorStoreFactory.create(
        store_type=settings.vector_store.type,
        collection_name=settings.vector_store.collection_name,
        persist_directory=settings.vector_store.persist_directory
    )
    
    # Get document count
    doc_count = vector_store.count()
    logger.info(f"Total documents in vector store: {doc_count}")
    
    # Test search
    if doc_count > 0:
        logger.info("Testing search functionality...")
        
        # Initialize embedding model for query
        embedding_model = EmbeddingFactory.create(
            model_type=settings.embedding.type,
            model_name=settings.embedding.model_name
        )
        
        # Test query
        test_query = "Who is Customer?"
        query_embedding = embedding_model.embed_text(test_query)
        
        results = vector_store.search(
            query_embedding=query_embedding,
            k=3
        )
        
        logger.info(f"Search results for '{test_query}':")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. Score: {result.score:.4f}")
            logger.info(f"   Document: {result.metadata.get('document_name', 'Unknown')}")
            logger.info(f"   Preview: {result.text[:100]}...")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Index PDF documents for RAG chatbot"
    )
    
    parser.add_argument(
        "pdf_directory",
        nargs="?",
        default="train-data",
        help="Directory containing PDF files (default: train-data)"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing vector store data before indexing"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify indexed documents after processing"
    )
    
    args = parser.parse_args()
    
    # Run indexing
    index_documents(
        pdf_directory=args.pdf_directory,
        config_path=args.config,
        clear_existing=args.clear
    )
    
    # Verify if requested
    if args.verify:
        verify_index(config_path=args.config)


if __name__ == "__main__":
    main()