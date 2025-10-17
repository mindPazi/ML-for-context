import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.search_engine import SearchEngine


@patch('src.search_engine.EmbeddingModel')
def test_init(mock_embedding_model):
    mock_model = MagicMock()
    mock_model.get_embedding_dim.return_value = 768
    mock_embedding_model.return_value = mock_model
    
    engine = SearchEngine()
    
    assert engine.embedding_model == mock_model
    assert engine.vector_store.embedding_dim == 768


@patch('src.search_engine.EmbeddingModel')
def test_index_documents(mock_embedding_model):
    mock_model = MagicMock()
    mock_model.get_embedding_dim.return_value = 3
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mock_embedding_model.return_value = mock_model
    
    engine = SearchEngine()
    docs = ["doc1", "doc2"]
    
    count = engine.index_documents(docs)
    
    assert count == 2
    assert len(engine.vector_store.documents) == 2
    mock_model.encode.assert_called_once()


@patch('src.search_engine.EmbeddingModel')
def test_search_empty(mock_embedding_model):
    mock_model = MagicMock()
    mock_model.get_embedding_dim.return_value = 3
    mock_embedding_model.return_value = mock_model
    
    engine = SearchEngine()
    results = engine.search("query")
    
    assert results == []


@patch('src.search_engine.EmbeddingModel')
def test_clear(mock_embedding_model):
    mock_model = MagicMock()
    mock_model.get_embedding_dim.return_value = 3
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
    mock_embedding_model.return_value = mock_model
    
    engine = SearchEngine()
    engine.index_documents(["doc1"])
    
    assert len(engine.vector_store.documents) == 1
    
    engine.clear()
    
    assert len(engine.vector_store.documents) == 0
    assert engine.vector_store.embeddings is None


@patch('src.search_engine.EmbeddingModel')
def test_info(mock_embedding_model):
    mock_model = MagicMock()
    mock_model.get_embedding_dim.return_value = 768
    mock_model.encode.return_value = np.array([[1.0] * 768])
    mock_embedding_model.return_value = mock_model
    
    engine = SearchEngine()
    
    info = engine.info()
    assert info["num_documents"] == 0
    assert info["embedding_dim"] == 768
    
    engine.index_documents(["doc1"])
    
    info = engine.info()
    assert info["num_documents"] == 1
    assert info["has_embeddings"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
