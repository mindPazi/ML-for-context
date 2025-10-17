import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search_engine import SearchEngine
from evaluation.cosqa_loader import CoSQALoader
from evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from bonus.extractor import extract_function_name
import json
from typing import Dict, List


def run_search(search_engine: SearchEngine, queries: List[Dict], top_k: int = 10):
    results = {}
    for query in queries:
        query_id = query["query_id"]
        query_text = query["query_text"]
        search_results = search_engine.search(query_text, top_k=top_k)
        retrieved_indices = [r["id"] for r in search_results]
        results[query_id] = retrieved_indices
    return results


def calculate_metrics(results: Dict, relevance: Dict):
    recalls = []
    mrrs = []
    ndcgs = []
    
    for query_id, retrieved in results.items():
        relevant = relevance[query_id]
        recalls.append(recall_at_k(relevant, retrieved, k=10))
        mrrs.append(mrr_at_k(relevant, retrieved, k=10))
        ndcgs.append(ndcg_at_k(relevant, retrieved, k=10))
    
    return {
        "recall@10": sum(recalls) / len(recalls),
        "mrr@10": sum(mrrs) / len(mrrs),
        "ndcg@10": sum(ndcgs) / len(ndcgs)
    }


def evaluate_with_function_names():
    print("\n" + "="*60)
    print("BONUS 1: Function Names vs Whole Bodies")
    print("="*60)
    
    model_path = "./models/unixcoder-finetuned/"
    
    print("\n[1/3] Loading CoSQA test dataset...")
    loader = CoSQALoader()
    corpus, queries, relevance = loader.load(split="test")
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)}")
    
    print("\n[2/3] Extracting function names from corpus...")
    corpus_names = [extract_function_name(code) for code in corpus]
    print(f"  Extracted {len(corpus_names)} function names")
    
    print("\n[3/3] Evaluating with function names...")
    engine_names = SearchEngine(model_name=model_path)
    engine_names.index_documents(corpus_names, show_progress=True)
    results_names = run_search(engine_names, queries)
    metrics_names = calculate_metrics(results_names, relevance)
    print(f"  RECALL@10: {metrics_names['recall@10']:.4f}")
    print(f"  MRR@10: {metrics_names['mrr@10']:.4f}")
    print(f"  NDCG@10: {metrics_names['ndcg@10']:.4f}")
    
    os.makedirs("bonus/results", exist_ok=True)
    
    with open("bonus/results/function_names.json", "w") as f:
        json.dump(metrics_names, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  RECALL@10: {metrics_names['recall@10']:.4f}")
    print(f"  MRR@10:    {metrics_names['mrr@10']:.4f}")
    print(f"  NDCG@10:   {metrics_names['ndcg@10']:.4f}")
    print(f"\nSaved to: bonus/results/function_names.json")
    print("="*60)


if __name__ == "__main__":
    evaluate_with_function_names()
