import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        try:
            self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=5.0)
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant: {e}")
            self.client = None
            
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # CrossEncoder disabled due to PyTorch compatibility issues
        self.reranker = None

    def search(self, query: str, k: int = 10) -> List[Document]:
        """Performs vector search (re-ranking temporarily disabled)."""
        if not self.client:
            print("Qdrant client not initialized.")
            return []
            
        query_vector = self.model.encode(query).tolist()
        
        try:
            # Retrieve top k candidates using dense vector search
            from qdrant_client.models import PointStruct, VectorParams, Distance
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k
            ).points
        except Exception as e:
            print(f"Search failed: {e}")
            return []
        
        if not results:
            return []
            
        # Convert to documents
        documents = []
        for res in results:
            text = res.payload.get("text", "")
            documents.append(Document(
                page_content=text,
                metadata=res.payload
            ))
        
        return documents

class MultiSourceRetriever:
    def __init__(self):
        self.wiki_retriever = HybridRetriever("wiki_rag")
        self.arxiv_retriever = HybridRetriever("arxiv_rag")
        
    def retrieve(self, query: str, source: str = "all") -> List[Document]:
        docs = []
        if source in ["all", "wiki"]:
            try:
                docs.extend(self.wiki_retriever.search(query))
            except Exception as e:
                print(f"Wiki retrieval failed: {e}")
                
        if source in ["all", "arxiv"]:
            try:
                docs.extend(self.arxiv_retriever.search(query))
            except Exception as e:
                print(f"ArXiv retrieval failed: {e}")
                
        return docs
