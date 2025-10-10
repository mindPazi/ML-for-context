import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.vector_store import VectorStore


def test_init():
    store = VectorStore()
    assert store.collection_name == "documents"
    assert store.embedding_dim == 768
    assert store.documents == []
    assert store.embeddings is None
    assert store.metadata == []
    assert store.ids == []


def test_add_documents():
    store = VectorStore(embedding_dim=3)
    
    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    metadata = [{"id": 1}, {"id": 2}, {"id": 3}]
    
    count = store.add_documents(docs, embeddings, metadata)
    
    assert count == 3
    assert len(store.documents) == 3
    assert store.documents == docs
    assert store.embeddings.shape == (3, 3)
    assert len(store.ids) == 3
    assert len(store.metadata) == 3
    assert store.metadata == metadata


def test_add_documents_without_metadata():
    store = VectorStore(embedding_dim=2)
    
    docs = ["doc1", "doc2"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    count = store.add_documents(docs, embeddings)
    
    assert count == 2
    assert len(store.metadata) == 2
    assert store.metadata == [{}, {}]




def test_delete_collection():
    store = VectorStore(embedding_dim=2)
    
    docs = ["doc1", "doc2"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    store.add_documents(docs, embeddings)
    
    assert len(store.documents) == 2
    
    store.delete_collection()
    
    assert store.documents == []
    assert store.embeddings is None
    assert store.metadata == []
    assert store.ids == []


def test_get_collection_info():
    store = VectorStore(collection_name="test", embedding_dim=512)
    
    info = store.get_collection_info()
    assert info["collection_name"] == "test"
    assert info["num_documents"] == 0
    assert info["embedding_dim"] == 512
    assert info["has_embeddings"] == False
    
    docs = ["doc1"]
    embeddings = np.array([[1.0] * 512])
    store.add_documents(docs, embeddings)
    
    info = store.get_collection_info()
    assert info["num_documents"] == 1
    assert info["has_embeddings"] == True


def test_multiple_add_documents():
    store = VectorStore(embedding_dim=2)
    
    docs1 = ["doc1", "doc2"]
    embeddings1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    store.add_documents(docs1, embeddings1)
    
    docs2 = ["doc3", "doc4"]
    embeddings2 = np.array([[0.5, 0.5], [0.7, 0.3]])
    store.add_documents(docs2, embeddings2)
    
    assert len(store.documents) == 4
    assert store.embeddings.shape == (4, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
