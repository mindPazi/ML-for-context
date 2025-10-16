from evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def test_recall_at_k():
    relevant = [0, 5]
    retrieved = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    result = recall_at_k(relevant, retrieved, k=10)
    assert result == 1.0
    
    result = recall_at_k(relevant, retrieved, k=1)
    assert result == 0.5


def test_mrr_at_k():
    relevant = [5]
    retrieved = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    result = mrr_at_k(relevant, retrieved, k=10)
    assert result == 1.0 / 6
    
    retrieved_first = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]
    result = mrr_at_k(relevant, retrieved_first, k=10)
    assert result == 1.0


def test_ndcg_at_k():
    relevant = [0, 1]
    retrieved = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    result = ndcg_at_k(relevant, retrieved, k=10)
    assert result > 0.0
    assert result <= 1.0
