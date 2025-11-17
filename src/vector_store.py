from typing import List, Dict, Any, Optional
import numpy as np
import uuid
import os


class VectorStore:

    def __init__(self, collection_name: str = "documents", embedding_dim: int = 768):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.ids = []

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> int:
        self.documents.extend(documents)

        new_ids = [str(uuid.uuid4()) for _ in documents]
        self.ids.extend(new_ids)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        return len(documents)

    def delete_collection(self):
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.ids = []

    def get_collection_info(self) -> Dict:
        return {
            "collection_name": self.collection_name,
            "num_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "has_embeddings": self.embeddings is not None,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            embeddings=self.embeddings,
            documents=self.documents,
            ids=self.ids,
            metadata=self.metadata,
        )

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.documents = data["documents"].tolist()
        self.ids = data["ids"].tolist()
        self.metadata = data["metadata"].tolist()
