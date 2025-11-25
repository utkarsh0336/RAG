import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from src.cache import RedisCache

_qdrant_client = None

class HybridRetriever:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_path = os.getenv("QDRANT_PATH")
        
        global _qdrant_client
        if _qdrant_client:
            self.client = _qdrant_client
        else:
            try:
                if qdrant_path:
                    self.client = QdrantClient(path=qdrant_path)
                elif os.path.exists("./qdrant_data"):
                    print("Found local qdrant_data, using it.")
                    self.client = QdrantClient(path="./qdrant_data")
                elif qdrant_url == ":memory:":
                    self.client = QdrantClient(location=":memory:")
                else:
                    self.client = QdrantClient(url=qdrant_url)
                _qdrant_client = self.client
            except Exception as e:
                print(f"Failed to connect to Qdrant: {e}")
                self.client = None
            
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize sparse model
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        # Re-enable CrossEncoder for better re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize cache
        try:
            self.cache = RedisCache()
        except Exception as e:
            print(f"Redis cache not available: {e}")
            self.cache = None

    def search(self, query: str, k: int = 10) -> List[Document]:
        """Performs hybrid search with dense retrieval + CrossEncoder re-ranking + vector caching."""
        if not self.client:
            print("Qdrant client not initialized.")
            return []
        
        # Try to get cached vector first (Tier 2)
        query_vector = None
        if self.cache:
            query_vector = self.cache.get_vector(query)
            if query_vector:
                print(f"✓ Cache hit: vector for '{query[:30]}...'")
        
        # If not cached, encode and cache it
        if query_vector is None:
            query_vector = self.model.encode(query).tolist()
            if self.cache:
                self.cache.set_vector(query, query_vector)
        
        try:
            # 1. Retrieve top 2*k candidates using hybrid search
            # 1. Retrieve top 2*k candidates using hybrid search
            from qdrant_client.models import PointStruct, VectorParams, Distance, Prefetch, SparseVector
            
            # Generate sparse vector
            # fastembed returns a generator of SparseEmbedding (indices, values)
            sparse_embedding = list(self.sparse_model.embed(query))[0]
            
            # Hybrid search with prefetch
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_embedding.indices.tolist(),
                            values=sparse_embedding.values.tolist()
                        ),
                        using="sparse",
                        limit=k * 2
                    )
                ],
                query=query_vector,
                using="dense",
                limit=k * 2  # Retrieve more candidates for re-ranking
            ).points
        except Exception as e:
            print(f"Hybrid search failed: {e}")
            results = []
        
        # Fallback to dense search if hybrid returned nothing
        if not results:
            print("Hybrid search returned no results. Falling back to dense search.")
            try:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using="dense",
                    limit=k * 2
                ).points
            except Exception as e:
                print(f"Dense search failed: {e}")
                return []
        
        if not results:
            return []
            
        # 2. Re-rank results using CrossEncoder
        documents = []
        passages = []
        for res in results:
            text = res.payload.get("text", "")
            passages.append([query, text])
            documents.append(Document(
                page_content=text,
                metadata=res.payload
            ))
            
        # Score all query-document pairs
        scores = self.reranker.predict(passages)
        
        # Sort by re-ranking score (higher is better)
        ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        # Return top k after re-ranking
        return [doc for doc, score in ranked_results[:k]]

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
