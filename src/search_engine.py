from typing import List, Dict, Any, Optional
import numpy as np
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class SearchEngine:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(embedding_dim=self.embedding_model.get_embedding_dim())
    
    def index_documents(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> int:
        embeddings = self.embedding_model.encode(
            documents, 
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        return self.vector_store.add_documents(documents, embeddings, metadata)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.vector_store.embeddings is None or len(self.vector_store.embeddings) == 0:
            return []
        
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
        similarities = np.dot(self.vector_store.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = {
                "id": int(idx),
                "text": self.vector_store.documents[idx],
                "score": float(similarities[idx]),
                "metadata": self.vector_store.metadata[idx]
            }
            results.append(result)
        
        return results
    
    def clear(self):
        self.vector_store.delete_collection()
    
    def info(self) -> Dict:
        return self.vector_store.get_collection_info()
    
    def save(self, path: str):
        self.vector_store.save(path)
    
    def load(self, path: str):
        self.vector_store.load(path)
