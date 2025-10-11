import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api import app


@pytest.fixture
def client():
    with patch('src.api.SearchEngine') as MockSearchEngine:
        mock_engine = MagicMock()
        MockSearchEngine.return_value = mock_engine
        
        mock_engine.index_documents.return_value = 2
        mock_engine.search.return_value = [
            {'text': 'doc1', 'score': 0.9},
            {'text': 'doc2', 'score': 0.8}
        ]
        mock_engine.info.return_value = {
            'collection_name': 'documents',
            'num_documents': 2,
            'embedding_dim': 768,
            'has_embeddings': True
        }
        
        yield TestClient(app)

def test_index(client):
    response = client.post('/index', json={
        'documents': ['doc1', 'doc2']
    })
    assert response.status_code == 200
    assert response.json() == {'indexed': 2}

def test_search(client):
    response = client.post('/search', json={
        'query': 'test',
        'top_k': 5
    })
    assert response.status_code == 200
    data = response.json()
    assert 'results' in data
    assert len(data['results']) == 2

def test_info(client):
    response = client.get('/info')
    assert response.status_code == 200
    data = response.json()
    assert data['collection_name'] == 'documents'
    assert data['num_documents'] == 2

def test_clear(client):
    response = client.delete('/clear')
    assert response.status_code == 200
    assert response.json() == {'status': 'cleared'}
