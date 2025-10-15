import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os
import json
import torch
import pickle
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_engine import SearchEngine
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from evaluation.cosqa_loader import CoSQALoader


class SearchEngineWithMetric(SearchEngine):
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_seq_length: int = 256,
        device: Optional[str] = None,
        similarity_metric: str = "cosine",
        normalize: bool = True,
        embedding_model = None
    ):
        self.similarity_metric = similarity_metric
        self.normalize = normalize
        
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = EmbeddingModel(
                model_name=model_name,
                max_seq_length=max_seq_length,
                device=device
            )
        
        self.vector_store = VectorStore(embedding_dim=self.embedding_model.get_embedding_dim())
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.vector_store.embeddings is None or len(self.vector_store.embeddings) == 0:
            return []
        
        query_embedding = self.embedding_model.encode(query, normalize=self.normalize)
        
        if self.similarity_metric == "cosine":
            if self.normalize:
                # For normalized embeddings, cosine similarity = dot product
                scores = np.dot(self.vector_store.embeddings, query_embedding)
            else:
                # For unnormalized embeddings, compute cosine similarity manually
                doc_norms = np.linalg.norm(self.vector_store.embeddings, axis=1)
                query_norm = np.linalg.norm(query_embedding)
                scores = np.dot(self.vector_store.embeddings, query_embedding) / (doc_norms * query_norm)
        elif self.similarity_metric == "euclidean":
            distances = np.linalg.norm(self.vector_store.embeddings - query_embedding, axis=1)
            scores = 1 / (1 + distances)
        elif self.similarity_metric == "manhattan":
            distances = np.sum(np.abs(self.vector_store.embeddings - query_embedding), axis=1)
            scores = 1 / (1 + distances)
        elif self.similarity_metric == "dot_product":
            scores = np.dot(self.vector_store.embeddings, query_embedding)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = {
                "id": int(idx),
                "text": self.vector_store.documents[idx],
                "score": float(scores[idx]),
                "metadata": self.vector_store.metadata[idx]
            }
            results.append(result)
        
        return results


class MetricsEmbeddingModel:
    def __init__(self, model_name: str, max_seq_length: int, device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = torch.device(device if device else "mps")
        self.model.to(self.device)
        self.model.max_seq_length = max_seq_length
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False, normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


_model_cache = {}
CACHE_DIR = 'cache'


def get_or_create_model(model_path: str):
    if 'model' not in _model_cache:
        model = MetricsEmbeddingModel(model_name=model_path, max_seq_length=256, device=None)
        _model_cache['model'] = model
    return _model_cache['model']


def get_or_create_embeddings(corpus: List[str], model_path: str, normalize: bool) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    norm_str = "normalized" if normalize else "unnormalized"
    cache_file = os.path.join(CACHE_DIR, f'embeddings_{norm_str}.pkl')
    
    if os.path.exists(cache_file):
        print(f"      Loading embeddings from disk cache for normalize={normalize}...")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"      ✓ Loaded {len(embeddings)} embeddings from cache")
    else:
        print(f"      Computing embeddings for normalize={normalize}...")
        model = get_or_create_model(model_path)
        embeddings = model.encode(corpus, batch_size=32, show_progress_bar=True, normalize=normalize)
        
        print(f"      Saving embeddings to disk cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"      ✓ Cached {len(embeddings)} embeddings to {cache_file}")
    
    return embeddings


def run_search(engine: SearchEngineWithMetric, queries: List[Dict], top_k: int = 10) -> Dict[str, List[int]]:
    results = {}
    for i, query in enumerate(queries):
        query_id = query["query_id"]
        query_text = query["query_text"]
        search_results = engine.search(query_text, top_k=top_k)
        retrieved_indices = [r["id"] for r in search_results]
        results[query_id] = retrieved_indices
    return results


def calculate_metrics(results: Dict[str, List[int]], relevance: Dict[str, List[int]]):
    recalls = []
    mrrs = []
    ndcgs = []
    
    for query_id, retrieved in results.items():
        relevant = relevance[query_id]
        
        pred_set = set(retrieved[:10])
        truth_set = set(relevant)
        recalls.append(len(pred_set & truth_set) / len(truth_set) if truth_set else 0)
        
        for i, doc_idx in enumerate(retrieved[:10], 1):
            if doc_idx in truth_set:
                mrrs.append(1.0 / i)
                break
        else:
            mrrs.append(0.0)
        
        dcg = 0
        for i, doc_idx in enumerate(retrieved[:10], 1):
            if doc_idx in truth_set:
                dcg += 1.0 / np.log2(i + 1)
        
        idcg = 0
        for i in range(1, min(len(truth_set), 10) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    return {
        "recall@10": sum(recalls) / len(recalls),
        "mrr@10": sum(mrrs) / len(mrrs),
        "ndcg@10": sum(ndcgs) / len(ndcgs)
    }


def evaluate(metric_name: str, normalize: bool = True, model_path: str = "./models/unixcoder-finetuned"):
    norm_str = "normalized" if normalize else "unnormalized"
    print(f"\n{'='*60}")
    print(f"Evaluating: {metric_name} ({norm_str})")
    print(f"{'='*60}")
    
    loader = CoSQALoader()
    corpus, queries, relevance = loader.load(split='test')
    
    print(f"Loading model from {model_path}")
    
    model = get_or_create_model(model_path)
    
    engine = SearchEngineWithMetric(
        model_name=model_path,
        max_seq_length=256,
        similarity_metric=metric_name,
        normalize=normalize,
        embedding_model=model
    )
    
    print(f"Indexing {len(corpus)} documents...")
    corpus_embeddings = get_or_create_embeddings(corpus, model_path, normalize)
    engine.vector_store.embeddings = corpus_embeddings
    engine.vector_store.documents = corpus
    engine.vector_store.metadata = [{"doc_id": i} for i in range(len(corpus))]
    
    print(f"Running {len(queries)} queries...")
    results = run_search(engine, queries, top_k=10)
    
    metrics = calculate_metrics(results, relevance)
    
    print(f"\nResults for {metric_name} ({norm_str}):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics


def run_all_metrics():
    metrics_list = ['cosine', 'euclidean', 'manhattan', 'dot_product']
    normalize_options = [True, False]
    results = {}
    
    for normalize in normalize_options:
        for metric in metrics_list:
            norm_str = "normalized" if normalize else "unnormalized"
            key = f"{metric}_{norm_str}"
            try:
                metrics = evaluate(metric, normalize=normalize)
                results[key] = metrics
            except Exception as e:
                print(f"Error evaluating {key}: {e}")
                results[key] = {"error": str(e)}
    
    os.makedirs('bonus/results', exist_ok=True)
    output_path = 'bonus/results/similarity_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Summary of all metrics:")
    print(f"{'='*60}")
    
    print("\nNORMALIZED EMBEDDINGS:")
    for metric in metrics_list:
        key = f"{metric}_normalized"
        if key in results and 'error' not in results[key]:
            print(f"\n{metric.upper()}:")
            for k, v in results[key].items():
                print(f"  {k}: {v:.4f}")
    
    print("\n" + "="*60)
    print("UNNORMALIZED EMBEDDINGS:")
    for metric in metrics_list:
        key = f"{metric}_unnormalized"
        if key in results and 'error' not in results[key]:
            print(f"\n{metric.upper()}:")
            for k, v in results[key].items():
                print(f"  {k}: {v:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_all_metrics()
