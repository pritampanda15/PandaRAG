"""
Vector store for ADMET knowledge retrieval.

Uses ChromaDB for efficient similarity search with
sentence-transformer embeddings.
"""

from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .document_processor import Document


class VectorStore:
    """
    Vector database for ADMET knowledge retrieval.
    
    Stores document embeddings in ChromaDB and provides
    semantic search capabilities.
    
    Attributes:
        collection_name: Name of the ChromaDB collection.
        embedding_model: Sentence transformer model name.
    """
    
    def __init__(
        self,
        collection_name: str = "admet_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name for the document collection.
            embedding_model: Sentence transformer model identifier.
            persist_directory: Directory to persist ChromaDB data.
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        
        self.encoder = SentenceTransformer(embedding_model)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add.
            
        Returns:
            Number of documents added.
        """
        if not documents:
            return 0
        
        contents = [doc.content for doc in documents]
        embeddings = self.encoder.encode(contents).tolist()
        
        ids = [doc.doc_id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids)
            existing_ids = set(existing["ids"])
        except Exception:
            pass
        
        new_docs = [
            (doc, emb) for doc, emb in zip(documents, embeddings)
            if doc.doc_id not in existing_ids
        ]
        
        if new_docs:
            self.collection.add(
                ids=[doc.doc_id for doc, _ in new_docs],
                embeddings=[emb for _, emb in new_docs],
                documents=[doc.content for doc, _ in new_docs],
                metadatas=[doc.metadata for doc, _ in new_docs]
            )
        
        return len(new_docs)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filters.
            
        Returns:
            List of (content, metadata, score) tuples.
        """
        query_embedding = self.encoder.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=filter_metadata
        )
        
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        scores = [1 - d for d in distances]
        
        return list(zip(documents, metadatas, scores))
    
    def search_by_property(
        self,
        property_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search within documents for a specific ADMET property.
        
        Args:
            property_name: ADMET property to filter by.
            query: Search query text.
            top_k: Number of results.
            
        Returns:
            List of (content, metadata, score) tuples.
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter_metadata={"property": property_name}
        )
    
    def get_property_context(
        self,
        property_name: str,
        molecular_descriptors: Dict[str, float],
        top_k: int = 5
    ) -> str:
        """
        Build context for explaining a property prediction.
        
        Args:
            property_name: ADMET property being predicted.
            molecular_descriptors: Computed molecular features.
            top_k: Number of context chunks to retrieve.
            
        Returns:
            Combined context string for LLM prompt.
        """
        query = f"{property_name} prediction interpretation molecular properties"
        
        if "logp" in molecular_descriptors:
            query += f" lipophilicity LogP"
        if "tpsa" in molecular_descriptors:
            query += f" polar surface area permeability"
        if "molecular_weight" in molecular_descriptors:
            mw = molecular_descriptors["molecular_weight"]
            if mw > 500:
                query += " large molecule bioavailability"
        
        results = self.search(query, top_k=top_k)
        
        context_parts = []
        for content, metadata, score in results:
            source = metadata.get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection info.
        """
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "embedding_model": self.embedding_model_name,
        }
