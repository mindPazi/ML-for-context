import sys
import os
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.search_engine import SearchEngine
from src.evaluation.cosqa_loader import CoSQALoader
from src.evaluation.metrics import calculate_metrics
from typing import Dict, List
import logging
import argparse
import os
import hashlib
import pickle

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def get_cache_path(model_name: str, normalize: bool = True) -> str:
    cache_dir = "./cache/embeddings"
    os.makedirs(cache_dir, exist_ok=True)

    if "microsoft/unixcoder-base" in model_name:
        cache_file = f"{cache_dir}/embedding_base.pkl.npz"
    elif (
        "./models/unixcoder-finetuned" in model_name
        or "unixcoder-finetuned" in model_name
    ):
        norm_str = "normalized" if normalize else "notnormalized"
        cache_file = f"{cache_dir}/embeddings_finetuned_{norm_str}.pkl.npz"
    else:

        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        norm_str = "normalized" if normalize else "notnormalized"
        cache_file = f"{cache_dir}/embeddings_{model_hash}_{norm_str}.pkl.npz"

    return cache_file


def index_corpus(
    search_engine: SearchEngine,
    corpus: List[str],
    model_name: str,
    normalize: bool = True,
):
    cache_path = get_cache_path(model_name, len(corpus), normalize)

    if os.path.exists(cache_path):
        log.info(f"      Loading from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        search_engine.vector_store.embeddings = embeddings
        search_engine.vector_store.documents = corpus
        search_engine.vector_store.metadata = [
            {"doc_id": i} for i in range(len(corpus))
        ]
        log.info(f"      ✓ Loaded {len(corpus)} documents from cache")
    else:
        log.info(f"      Computing embeddings...")
        search_engine.index_documents(corpus, show_progress=True)

        log.info(f"      Saving to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(search_engine.vector_store.embeddings, f)
        log.info(f"      ✓ Cache saved")


def run(search_engine: SearchEngine, queries: List[Dict], top_k: int = 10):
    results = {}
    for query in queries:
        query_id = query["query_id"]
        query_text = query["query_text"]
        search_results = search_engine.search(query_text, top_k=top_k)
        retrieved_indices = [r["id"] for r in search_results]
        results[query_id] = retrieved_indices
    return results


def print_results(metrics: Dict):
    log.info("\n" + "=" * 50)
    log.info("EVALUATION RESULTS".center(50))
    log.info("=" * 50)
    for metric_name, value in metrics.items():
        log.info(f"  {metric_name.upper():<20} {value:.4f}")
    log.info("=" * 50)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate embedding model on CoSQA dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/unixcoder-base",
        help="Model to evaluate (base model name or path to fine-tuned model)",
    )
    parser.add_argument(
        "--test-subset",
        type=float,
        default=1.0,
        help="Fraction of test queries to evaluate (0.0-1.0). Default: 1.0 (100%%)",
    )
    args = parser.parse_args()

    log.info("\n" + "=" * 50)
    log.info("CoSQA EVALUATION PIPELINE".center(50))
    log.info("=" * 50)
    log.info(f"\nModel: {args.model}")

    log.info("\n[1/5] Loading dataset...")
    loader = CoSQALoader()
    corpus, queries, relevance = loader.load(split="test")
    log.info(f"      ├─ Corpus size: {len(corpus)}")
    log.info(f"      └─ Total queries: {len(queries)}")

    if args.test_subset < 1.0:

        random.seed(42)
        total_queries = len(queries)
        subset_size = int(total_queries * args.test_subset)
        random.shuffle(queries)
        queries = queries[:subset_size]
        query_ids = {q["query_id"] for q in queries}
        relevance = {qid: docs for qid, docs in relevance.items() if qid in query_ids}
        log.info(
            f"      └─ Using {args.test_subset*100:.0f}% of queries: {len(queries)}/{total_queries}"
        )

    log.info("\n[2/5] Initializing search engine...")
    search_engine = SearchEngine(model_name=args.model)

    log.info("\n[3/5] Indexing corpus...")

    index_corpus(search_engine, corpus, args.model, normalize=True)

    log.info("\n[4/5] Running queries...")
    results = run(search_engine, queries, top_k=10)

    log.info("\n[5/5] Calculating metrics...")
    metrics = calculate_metrics(results, relevance)

    print_results(metrics)


if __name__ == "__main__":
    main()
