"""PDF text extraction module for processing technical documentation."""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PDFDocument:
    """Represents a processed PDF document."""
    file_path: str
    file_name: str
    total_pages: int
    text_content: str
    page_contents: Dict[int, str]
    metadata: Dict[str, any]
    extraction_method: str


class PDFExtractor:
    """Handles PDF text extraction with multiple backend support."""
    
    def __init__(self, method: str = "auto"):
        """
        Initialize PDF extractor.
        
        Args:
            method: Extraction method - 'pypdf2', 'pdfplumber', 'pymupdf', or 'auto'
        """
        self.method = method
        self.supported_methods = {
            'pypdf2': self._extract_with_pypdf2,
            'pdfplumber': self._extract_with_pdfplumber,
            'pymupdf': self._extract_with_pymupdf
        }
    
    def extract_text(self, pdf_path: str) -> PDFDocument:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFDocument object containing extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Extracting text from: {pdf_path.name}")
        
        # Try extraction methods
        if self.method == "auto":
            # Try methods in order of preference
            for method_name, method_func in self.supported_methods.items():
                try:
                    doc = method_func(str(pdf_path))
                    # Post-process the extracted text
                    doc = self._post_process_document(doc)
                    return doc
                except Exception as e:
                    logger.warning(f"Method {method_name} failed: {str(e)}")
                    continue
            raise Exception("All extraction methods failed")
        else:
            # Use specific method
            if self.method not in self.supported_methods:
                raise ValueError(f"Unsupported method: {self.method}")
            doc = self.supported_methods[self.method](str(pdf_path))
            # Post-process the extracted text
            doc = self._post_process_document(doc)
            return doc
    
    def _extract_with_pypdf2(self, pdf_path: str) -> PDFDocument:
        """Extract text using PyPDF2."""
        page_contents = {}
        all_text = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            metadata = {
                'title': pdf_reader.metadata.get('/Title', ''),
                'author': pdf_reader.metadata.get('/Author', ''),
                'subject': pdf_reader.metadata.get('/Subject', ''),
                'creator': pdf_reader.metadata.get('/Creator', ''),
            }
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                page_contents[page_num + 1] = text
                all_text.append(text)
        
        return PDFDocument(
            file_path=pdf_path,
            file_name=os.path.basename(pdf_path),
            total_pages=total_pages,
            text_content='\n\n'.join(all_text),
            page_contents=page_contents,
            metadata=metadata,
            extraction_method='pypdf2'
        )
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> PDFDocument:
        """Extract text using pdfplumber (better for tables)."""
        page_contents = {}
        all_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            metadata = pdf.metadata or {}
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                
                # Also extract tables if present
                tables = page.extract_tables()
                if tables:
                    text += "\n\n[Tables found on this page]\n"
                    for table in tables:
                        # Convert table to text format
                        table_text = self._table_to_text(table)
                        text += table_text + "\n"
                
                page_contents[page_num] = text
                all_text.append(text)
        
        return PDFDocument(
            file_path=pdf_path,
            file_name=os.path.basename(pdf_path),
            total_pages=total_pages,
            text_content='\n\n'.join(all_text),
            page_contents=page_contents,
            metadata=metadata,
            extraction_method='pdfplumber'
        )
    
    def _extract_with_pymupdf(self, pdf_path: str) -> PDFDocument:
        """Extract text using PyMuPDF (fastest method)."""
        page_contents = {}
        all_text = []
        
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        metadata = pdf_document.metadata
        
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            text = page.get_text()
            page_contents[page_num + 1] = text
            all_text.append(text)
        
        pdf_document.close()
        
        return PDFDocument(
            file_path=pdf_path,
            file_name=os.path.basename(pdf_path),
            total_pages=total_pages,
            text_content='\n\n'.join(all_text),
            page_contents=page_contents,
            metadata=metadata,
            extraction_method='pymupdf'
        )
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to formatted text."""
        if not table:
            return ""
        
        # Calculate column widths
        col_widths = []
        for col_idx in range(len(table[0])):
            max_width = max(len(str(row[col_idx]) if row[col_idx] else "") 
                          for row in table)
            col_widths.append(max_width)
        
        # Format table
        lines = []
        for row in table:
            formatted_row = " | ".join(
                str(cell or "").ljust(col_widths[idx]) 
                for idx, cell in enumerate(row)
            )
            lines.append(formatted_row)
        
        return "\n".join(lines)
    
    def _post_process_document(self, doc: PDFDocument) -> PDFDocument:
        """
        Post-process extracted document to fix common issues.
        
        Args:
            doc: PDFDocument to process
            
        Returns:
            Processed PDFDocument
        """
        import re
        
        # Fix excessive spacing between characters
        # This pattern matches single characters separated by single spaces
        # but preserves normal word spacing
        def fix_char_spacing(text: str) -> str:
            # Pattern to match sequences like "C a m p a i g n" but not normal words
            # This looks for patterns where single characters are separated by single spaces
            lines = text.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Check if line has excessive character spacing
                # Count ratio of spaces to non-space characters
                if line.strip():
                    space_count = line.count(' ')
                    char_count = len(line.replace(' ', ''))
                    
                    # If there's roughly one space per character, it's likely spaced out
                    if char_count > 0 and space_count / char_count > 0.7:
                        # Remove spaces between single characters
                        # This regex matches single char followed by space and another single char
                        fixed_line = re.sub(r'(?<=\b\w)\s(?=\w\b)', '', line)
                        # Also handle cases at the beginning of words
                        fixed_line = re.sub(r'(\b\w)\s+(?=\w\s+\w)', r'\1', fixed_line)
                        # Clean up any remaining single character spacings
                        words = fixed_line.split()
                        new_words = []
                        i = 0
                        while i < len(words):
                            if i < len(words) - 1 and len(words[i]) == 1 and len(words[i+1]) == 1:
                                # Merge single characters
                                combined = words[i]
                                j = i + 1
                                while j < len(words) and len(words[j]) == 1:
                                    combined += words[j]
                                    j += 1
                                new_words.append(combined)
                                i = j
                            else:
                                new_words.append(words[i])
                                i += 1
                        fixed_line = ' '.join(new_words)
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
        
        # Process the main text content
        doc.text_content = fix_char_spacing(doc.text_content)
        
        # Process page contents
        for page_num, content in doc.page_contents.items():
            doc.page_contents[page_num] = fix_char_spacing(content)
        
        return doc
    
    def extract_from_directory(self, directory: str) -> List[PDFDocument]:
        """
        Extract text from all PDFs in a directory.
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of PDFDocument objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        documents = []
        for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
            try:
                doc = self.extract_text(str(pdf_file))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to extract {pdf_file.name}: {str(e)}")
                continue
        
        return documents


def main():
    """Test PDF extraction."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_file_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    extractor = PDFExtractor(method="auto")
    
    if os.path.isfile(path):
        # Extract single file
        doc = extractor.extract_text(path)
        print(f"Extracted {doc.total_pages} pages from {doc.file_name}")
        print(f"Text preview: {doc.text_content[:500]}...")
    else:
        # Extract directory
        docs = extractor.extract_from_directory(path)
        print(f"Extracted {len(docs)} documents")
        for doc in docs:
            print(f"- {doc.file_name}: {doc.total_pages} pages")


if __name__ == "__main__":
    main()