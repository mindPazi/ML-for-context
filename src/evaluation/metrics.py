import numpy as np
from typing import List


def recall_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int = 10) -> float:
    if not relevant_docs:
        return 0.0
    retrieved_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    return len(retrieved_k & relevant_set) / len(relevant_set)


def mrr_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int = 10) -> float:
    relevant_set = set(relevant_docs)
    for rank, doc_id in enumerate(retrieved_docs[:k], 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int = 10) -> float:
    relevant_set = set(relevant_docs)
    dcg = sum(1.0 / np.log2(rank + 2) for rank, doc_id in enumerate(retrieved_docs[:k]) 
              if doc_id in relevant_set)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant_docs), k)))
    return dcg / idcg if idcg > 0 else 0.0
