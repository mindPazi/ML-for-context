import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os
import json
import torch
import pickle
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.search_engine import SearchEngine
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.evaluation.cosqa_loader import CoSQALoader
from src.evaluation.metrics import calculate_metrics, run_queries


class SearchEngineWithMetric(SearchEngine):
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_seq_length: int = 256,
        device: Optional[str] = None,
        similarity_metric: str = "cosine",
        normalize: bool = True,
        embedding_model=None,
    ):
        self.similarity_metric = similarity_metric
        self.normalize = normalize

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = EmbeddingModel(
                model_name=model_name, max_seq_length=max_seq_length, device=device
            )

        self.vector_store = VectorStore(
            embedding_dim=self.embedding_model.get_embedding_dim()
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if (
            self.vector_store.embeddings is None
            or len(self.vector_store.embeddings) == 0
        ):
            return []

        query_embedding = self.embedding_model.encode(query, normalize=self.normalize)

        if self.similarity_metric == "manhattan":
            distances = np.sum(
                np.abs(self.vector_store.embeddings - query_embedding), axis=1
            )
            top_indices = np.argsort(distances)[:top_k]
        elif self.similarity_metric == "euclidean":
            distances = np.linalg.norm(
                self.vector_store.embeddings - query_embedding, axis=1
            )
            top_indices = np.argsort(distances)[:top_k]
        elif self.similarity_metric == "cosine":
            if self.normalize:
                scores = np.dot(self.vector_store.embeddings, query_embedding)
            else:
                doc_norms = np.linalg.norm(self.vector_store.embeddings, axis=1)
                query_norm = np.linalg.norm(query_embedding)
                scores = np.dot(self.vector_store.embeddings, query_embedding) / (
                    doc_norms * query_norm
                )
            top_indices = np.argsort(scores)[-top_k:][::-1]
        elif self.similarity_metric == "dot_product":
            scores = np.dot(self.vector_store.embeddings, query_embedding)
            top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if self.similarity_metric in ["euclidean", "manhattan"]:
                score_value = float(distances[idx])
            else:
                score_value = float(scores[idx])

            result = {
                "id": int(idx),
                "text": self.vector_store.documents[idx],
                "score": score_value,
                "metadata": self.vector_store.metadata[idx],
            }
            results.append(result)

        return results


class MetricsEmbeddingModel:
    def __init__(
        self, model_name: str, max_seq_length: int, device: Optional[str] = None
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = torch.device(device if device else "mps")
        self.model.to(self.device)
        self.model.max_seq_length = max_seq_length
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


_model_cache = {}
CACHE_DIR = "cache"


def get_or_create_model(model_path: str):
    if "model" not in _model_cache:
        model = MetricsEmbeddingModel(
            model_name=model_path, max_seq_length=256, device=None
        )
        _model_cache["model"] = model
    return _model_cache["model"]


def get_or_create_embeddings(
    corpus: List[str], model_path: str, normalize: bool
) -> np.ndarray:
    cache_dir = "./cache/embeddings"
    os.makedirs(cache_dir, exist_ok=True)

    norm_str = "normalized" if normalize else "notnormalized"
    cache_file = os.path.join(cache_dir, f"embeddings_finetuned_{norm_str}.pkl.npz")

    if os.path.exists(cache_file):
        print(f"      Loading embeddings from disk cache for normalize={normalize}...")
        with open(cache_file, "rb") as f:
            embeddings = pickle.load(f)
        print(f"      ✓ Loaded {len(embeddings)} embeddings from cache")
    else:
        print(f"      Computing embeddings for normalize={normalize}...")
        model = get_or_create_model(model_path)
        embeddings = model.encode(
            corpus, batch_size=32, show_progress_bar=True, normalize=normalize
        )

        print(f"      Saving embeddings to disk cache...")
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"      ✓ Cached {len(embeddings)} embeddings to {cache_file}")

    return embeddings


def evaluate(
    metric_name: str,
    normalize: bool = True,
    model_path: str = "./models/unixcoder-finetuned",
):
    norm_str = "normalized" if normalize else "unnormalized"
    print(f"\n{'='*60}")
    print(f"Evaluating: {metric_name} ({norm_str})")
    print(f"{'='*60}")

    loader = CoSQALoader()
    corpus, queries, relevance = loader.load(split="test")

    print(f"Loading model from {model_path}")

    model = get_or_create_model(model_path)

    engine = SearchEngineWithMetric(
        model_name=model_path,
        max_seq_length=256,
        similarity_metric=metric_name,
        normalize=normalize,
        embedding_model=model,
    )

    print(f"Indexing {len(corpus)} documents...")
    corpus_embeddings = get_or_create_embeddings(corpus, model_path, normalize)
    engine.vector_store.embeddings = corpus_embeddings
    engine.vector_store.documents = corpus
    engine.vector_store.metadata = [{"doc_id": i} for i in range(len(corpus))]

    print(f"Running {len(queries)} queries...")
    results = run_queries(engine, queries, top_k=10)

    metrics = calculate_metrics(results, relevance)

    print(f"\nResults for {metric_name} ({norm_str}):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics


def run_all_metrics():
    metrics_list = ["cosine", "euclidean", "manhattan", "dot_product"]
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

    os.makedirs("src/bonus/results", exist_ok=True)
    output_path = "src/bonus/results/similarity_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary of all metrics:")
    print(f"{'='*60}")

    print("\nNORMALIZED EMBEDDINGS:")
    for metric in metrics_list:
        key = f"{metric}_normalized"
        if key in results and "error" not in results[key]:
            print(f"\n{metric.upper()}:")
            for k, v in results[key].items():
                print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("UNNORMALIZED EMBEDDINGS:")
    for metric in metrics_list:
        key = f"{metric}_unnormalized"
        if key in results and "error" not in results[key]:
            print(f"\n{metric.upper()}:")
            for k, v in results[key].items():
                print(f"  {k}: {v:.4f}")

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_all_metrics()
