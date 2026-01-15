import os
from pathlib import Path
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """Manages the RAG pipeline for personalized LLM."""

    def __init__(self, chroma_db_path: str = None, embedding_model: str = None):
        """
        Initialize RAG pipeline.
        
        Args:
            chroma_db_path: Path to ChromaDB directory
            embedding_model: Sentence transformer model name
        """
        self.chroma_db_path = chroma_db_path or os.getenv(
            'CHROMA_DB_PATH', './chroma_db'
        )
        self.embedding_model_name = embedding_model or os.getenv(
            'EMBEDDING_MODEL', 'all-MiniLM-L6-v2'
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        
        # Create directory if it doesn't exist
        Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with new API
        self.client = chromadb.PersistentClient(
            path=self.chroma_db_path
        )
        
        self.collection_name = "whatsapp_messages"
    
    def load_and_index(self, documents: List[str], chunk_size: int = None, 
                       chunk_overlap: int = None) -> None:
        """
        Load documents and create embeddings.
        
        Args:
            documents: List of text chunks to index
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', 500))
        chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', 100))
        
        print(f"Indexing {len(documents)} documents...")
        
        # Create or get existing collection
        try:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Creating new collection: {e}")
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            chunks.extend(text_splitter.split_text(doc))
        
        print(f"Created {len(chunks)} chunks from documents")
        
        # Generate embeddings and add to ChromaDB
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk)
            collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": "whatsapp"}]
            )
        
        print(f"Indexed {len(chunks)} chunks successfully")
    
    def query(self, query_text: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query_text: The query string
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        collection = self.client.get_collection(self.collection_name)
        
        # Generate embedding for query
        query_embedding = self.embeddings.embed_query(query_text)
        
        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        if results['documents']:
            return results['documents'][0]
        return []
    
    def get_context(self, query_text: str, k: int = 5) -> str:
        """
        Get formatted context for LLM.
        
        Args:
            query_text: The query string
            k: Number of relevant chunks to include
            
        Returns:
            Formatted context string
        """
        relevant_docs = self.query(query_text, k=k)
        
        if not relevant_docs:
            return "No relevant context found."
        
        context = "Relevant messages from your chat history:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"[{i}] {doc}\n\n"
        
        return context
