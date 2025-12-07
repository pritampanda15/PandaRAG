"""
Document processing for ADMET knowledge base construction.

Handles ingestion of pharmacokinetic literature, drug labels,
and scientific text for the RAG pipeline.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
import hashlib
from pypdf import PdfReader


@dataclass
class Document:
    """
    Represents a processed document chunk.
    
    Attributes:
        content: Text content of the chunk.
        metadata: Dictionary of document metadata.
        doc_id: Unique identifier for the chunk.
    """
    content: str
    metadata: Dict[str, str]
    doc_id: str


class DocumentProcessor:
    """
    Processes documents for the ADMET knowledge base.
    
    Handles PDF and text files, chunking them into appropriate
    sizes for embedding and retrieval.
    
    Attributes:
        chunk_size: Target size for document chunks.
        chunk_overlap: Number of characters to overlap between chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _generate_doc_id(self, content: str, source: str) -> str:
        """Generate unique document ID from content hash."""
        hash_input = f"{source}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _chunk_text(
        self,
        text: str,
        source: str,
        metadata: Dict[str, str]
    ) -> List[Document]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full document text.
            source: Source file path.
            metadata: Document metadata.
            
        Returns:
            List of Document chunks.
        """
        chunks = []
        text = text.strip()
        
        if len(text) <= self.chunk_size:
            doc_id = self._generate_doc_id(text, source)
            return [Document(
                content=text,
                metadata={**metadata, "chunk": 0},
                doc_id=doc_id
            )]
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                break_point = text.rfind(". ", start, end)
                if break_point > start:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                doc_id = self._generate_doc_id(chunk_text, source)
                chunks.append(Document(
                    content=chunk_text,
                    metadata={**metadata, "chunk": chunk_idx},
                    doc_id=doc_id
                ))
                chunk_idx += 1
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Extract and chunk text from a PDF file.
        
        Args:
            file_path: Path to PDF file.
            
        Returns:
            List of Document chunks.
        """
        path = Path(file_path)
        
        reader = PdfReader(file_path)
        
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        
        combined_text = "\n\n".join(full_text)
        
        metadata = {
            "source": path.name,
            "file_type": "pdf",
            "num_pages": len(reader.pages),
        }
        
        return self._chunk_text(combined_text, file_path, metadata)
    
    def process_text(self, file_path: str) -> List[Document]:
        """
        Process a plain text file.
        
        Args:
            file_path: Path to text file.
            
        Returns:
            List of Document chunks.
        """
        path = Path(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        metadata = {
            "source": path.name,
            "file_type": "text",
        }
        
        return self._chunk_text(text, file_path, metadata)
    
    def process_markdown(self, file_path: str) -> List[Document]:
        """
        Process a markdown file with section awareness.
        
        Args:
            file_path: Path to markdown file.
            
        Returns:
            List of Document chunks.
        """
        path = Path(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        sections = []
        current_section = ""
        current_header = ""
        
        for line in text.split("\n"):
            if line.startswith("#"):
                if current_section:
                    sections.append((current_header, current_section))
                current_header = line.strip("# ").strip()
                current_section = ""
            else:
                current_section += line + "\n"
        
        if current_section:
            sections.append((current_header, current_section))
        
        documents = []
        for header, content in sections:
            metadata = {
                "source": path.name,
                "file_type": "markdown",
                "section": header,
            }
            documents.extend(self._chunk_text(content, file_path, metadata))
        
        return documents
    
    def process_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None
    ) -> Generator[Document, None, None]:
        """
        Process all supported files in a directory.
        
        Args:
            directory: Path to directory.
            extensions: List of file extensions to process.
            
        Yields:
            Document chunks from all files.
        """
        if extensions is None:
            extensions = [".pdf", ".txt", ".md"]
        
        dir_path = Path(directory)
        
        for ext in extensions:
            for file_path in dir_path.glob(f"**/*{ext}"):
                print(f"Processing: {file_path.name}")
                
                try:
                    if ext == ".pdf":
                        docs = self.process_pdf(str(file_path))
                    elif ext == ".md":
                        docs = self.process_markdown(str(file_path))
                    else:
                        docs = self.process_text(str(file_path))
                    
                    for doc in docs:
                        yield doc
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
    
    def create_admet_document(
        self,
        property_name: str,
        description: str,
        interpretation_guide: str,
        thresholds: Optional[Dict[str, float]] = None,
        references: Optional[List[str]] = None
    ) -> Document:
        """
        Create a structured ADMET property document.
        
        Args:
            property_name: Name of the ADMET property.
            description: Description of what the property measures.
            interpretation_guide: How to interpret prediction values.
            thresholds: Optional threshold values for classification.
            references: Optional list of reference sources.
            
        Returns:
            Document object for the ADMET property.
        """
        content_parts = [
            f"ADMET Property: {property_name}",
            f"\nDescription: {description}",
            f"\nInterpretation: {interpretation_guide}",
        ]
        
        if thresholds:
            threshold_str = ", ".join(
                f"{k}: {v}" for k, v in thresholds.items()
            )
            content_parts.append(f"\nThresholds: {threshold_str}")
        
        if references:
            ref_str = "\n".join(f"- {ref}" for ref in references)
            content_parts.append(f"\nReferences:\n{ref_str}")
        
        content = "\n".join(content_parts)
        
        metadata = {
            "source": "admet_knowledge_base",
            "property": property_name,
            "document_type": "property_definition",
        }
        
        doc_id = self._generate_doc_id(content, property_name)
        
        return Document(content=content, metadata=metadata, doc_id=doc_id)
