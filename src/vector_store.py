import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import shutil
from typing import List, Dict, Any
from src.config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

class CustomEmbeddings:
    """Custom embedding wrapper for SentenceTransformer"""
    def __init__(self, model_name: str):
        # Explicitly force CPU and avoid accelerate's device_map if possible
        self.model = SentenceTransformer(model_name, device='cpu', trust_remote_code=True)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

class VectorStoreManager:
    def __init__(self):
        self.embedding_function = CustomEmbeddings(EMBEDDING_MODEL_NAME)
        self.persist_directory = CHROMA_DB_DIR
        
        # Initialize Chroma
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
            collection_name=COLLECTION_NAME
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

    def add_document(self, filename: str, text: str) -> int:
        """
        Chunks and adds a document to the vector store.
        Returns the number of chunks added.
        """
        print(f"\n[VectorStore] add_document called for: {filename}")
        print(f"[VectorStore] Text length: {len(text)}")
        
        if not text:
            print(f"[VectorStore] No text provided, returning 0")
            return 0
            
        # Check if document already exists and delete it to prevent duplicates
        print(f"[VectorStore] Checking for existing chunks of {filename}...")
        self.delete_documents([filename])
            
        # Create chunks
        chunks = self.text_splitter.split_text(text)
        print(f"[VectorStore] Created {len(chunks)} chunks")
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_id": i
                }
            )
            documents.append(doc)
            
        print(f"[VectorStore] Adding {len(documents)} documents to Chroma")
        self.vector_db.add_documents(documents)
        print(f"[VectorStore] Successfully added documents for {filename}")
        # Chroma 0.4+ persists automatically, but explicit persist calls are deprecated in newer versions.
        # If using older langchain/chroma versions, might need self.vector_db.persist()
        return len(chunks)

    def query_similarity(self, query: str, k: int = 5) -> List[Document]:
        """Queries the vector store for similar documents."""
        print(f"\n[VectorStore] query_similarity called")
        print(f"[VectorStore] Query: {query}")
        print(f"[VectorStore] k: {k}")
        
        results = self.vector_db.similarity_search(query, k=k)
        
        print(f"[VectorStore] Found {len(results)} results")
        
        # Deduplicate results based on page_content
        unique_results = []
        seen_content = set()
        
        for doc in results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
                
        print(f"[VectorStore] Unique results: {len(unique_results)}")
        for i, doc in enumerate(unique_results):
            print(f"[VectorStore] Result {i+1}: source={doc.metadata.get('source', 'unknown')}, chunk_id={doc.metadata.get('chunk_id', 'N/A')}")
        
        return unique_results
    
    def query_similarity_filtered(self, query: str, source_filter: List[str] = None, k: int = 5) -> List[Document]:
        """Queries the vector store, optionally filtering by source filenames."""
        print(f"\n[VectorStore] query_similarity_filtered called")
        print(f"[VectorStore] Query: {query}")
        print(f"[VectorStore] Source filter: {source_filter}")
        print(f"[VectorStore] k: {k}")
        
        if source_filter:
            # Query with metadata filter
            results = self.vector_db.similarity_search(
                query, 
                k=k,
                filter={"source": {"$in": source_filter}}
            )
        else:
            results = self.vector_db.similarity_search(query, k=k)
        
        print(f"[VectorStore] Found {len(results)} results")
        
        # Deduplicate results based on page_content
        unique_results = []
        seen_content = set()
        
        for doc in results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
                
        print(f"[VectorStore] Unique results: {len(unique_results)}")
        for i, doc in enumerate(unique_results):
            print(f"[VectorStore] Result {i+1}: source={doc.metadata.get('source', 'unknown')}, chunk_id={doc.metadata.get('chunk_id', 'N/A')}")
        
        return unique_results

    def list_documents(self) -> List[str]:
        """
        Returns a list of unique source filenames in the DB.
        Lists all unique document sources in the vector store.
        """
        try:
            # Get all documents
            all_docs = self.vector_db.get()
            if all_docs and 'metadatas' in all_docs:
                sources = set([meta.get('source', 'unknown') for meta in all_docs['metadatas']])
                return sorted(list(sources))
        except Exception as e:
            print(f"Error listing documents: {e}")
        return []
    
    def delete_documents(self, filenames: List[str]):
        """Delete all chunks from specific documents by filename."""
        print(f"\n[VectorStore] delete_documents called for: {filenames}")
        
        try:
            # Get all documents
            all_docs = self.vector_db.get()
            
            if all_docs and 'ids' in all_docs and 'metadatas' in all_docs:
                # Find IDs of documents to delete
                ids_to_delete = []
                for i, meta in enumerate(all_docs['metadatas']):
                    if meta.get('source') in filenames:
                        ids_to_delete.append(all_docs['ids'][i])
                
                if ids_to_delete:
                    print(f"[VectorStore] Deleting {len(ids_to_delete)} chunks")
                    self.vector_db.delete(ids=ids_to_delete)
                    print(f"[VectorStore] Successfully deleted documents: {filenames}")
                else:
                    print(f"[VectorStore] No chunks found for: {filenames}")
        except Exception as e:
            print(f"[VectorStore] Error deleting documents: {e}")

    def reset_db(self):
        """
        Clears the database.
        """
        # Delete the collection and re-create
        try:
            self.vector_db.delete_collection()
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=COLLECTION_NAME
            )
        except Exception as e:
            print(f"Error resetting DB: {e}")
