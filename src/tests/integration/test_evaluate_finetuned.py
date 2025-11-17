from src.search_engine import SearchEngine
from src.evaluation.cosqa_loader import CoSQALoader
from src.evaluation.metrics import calculate_metrics
from src.training.config import TrainingConfig
from typing import Dict, List
import os


def print_header(title: str):
    print("\n")
    print("=" * 50)
    print(title.center(50))
    print("=" * 50)
    print()


def index_corpus(search_engine: SearchEngine, corpus: List[str], cache_path: str = None):
    if cache_path and os.path.exists(cache_path):
        print(f"      Loading cached embeddings from {cache_path}...")
        search_engine.load(cache_path)
    else:
        search_engine.index_documents(corpus, show_progress=True)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            search_engine.save(cache_path)
            print(f"      Saved embeddings to {cache_path}")


def run(search_engine: SearchEngine, queries: List[Dict], top_k: int = 10):
    results = {}
    for query in queries:
        query_id = query["query_id"]
        query_text = query["query_text"]
        search_results = search_engine.search(query_text, top_k=top_k)
        retrieved_indices = [r["id"] for r in search_results]
        results[query_id] = retrieved_indices
    return results


def print_results(metrics: Dict, model_name: str):
    print(f"\n{'EVALUATION RESULTS - ' + model_name:^50}")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper():<20} {value:.4f}")
    print("=" * 50)


def main():
    print_header("FINE-TUNED MODEL EVALUATION TEST")
    
    finetuned_model_path = "./models/unixcoder-finetuned"
    if not os.path.exists(finetuned_model_path):
        print(f"Fine-tuned model not found at: {finetuned_model_path}")
        print("\nPlease train the model first:")
        print("  python -m training.train")
        return
    
    print("Fine-tuned model found")
    
    print("\n[1/6] Loading dataset...")
    loader = CoSQALoader()
    corpus, queries, relevance = loader.load(split="test")
    print(f"      Corpus size: {len(corpus)}")
    print(f"      Queries: {len(queries)}")
    
    print("\n[2/6] Evaluating base model...")
    print("      Model: microsoft/unixcoder-base")
    base_search_engine = SearchEngine(model_name="microsoft/unixcoder-base")
    
    print("\n[3/6] Indexing corpus (base model)...")
    base_cache_path = "./cache/embeddings/base_model_test.pkl"
    index_corpus(base_search_engine, corpus, cache_path=base_cache_path)
    
    print("\n[4/6] Running queries (base model)...")
    base_results = run(base_search_engine, queries, top_k=10)
    base_metrics = calculate_metrics(base_results, relevance)
    
    print("\n[5/6] Evaluating fine-tuned model...")
    print(f"      Model: {finetuned_model_path}")
    finetuned_search_engine = SearchEngine(model_name=finetuned_model_path)
    
    print("\n[6/6] Indexing corpus (fine-tuned model)...")
    index_corpus(finetuned_search_engine, corpus)
    
    print("\nRunning queries (fine-tuned model)...")
    finetuned_results = run(finetuned_search_engine, queries, top_k=10)
    finetuned_metrics = calculate_metrics(finetuned_results, relevance)
    
    print_header("COMPARISON RESULTS")
    
    print("BASE MODEL (microsoft/unixcoder-base):")
    for metric_name, value in base_metrics.items():
        print(f"  {metric_name.upper():<20} {value:.4f}")
    
    print("\nFINE-TUNED MODEL:")
    for metric_name, value in finetuned_metrics.items():
        print(f"  {metric_name.upper():<20} {value:.4f}")
    
    print("\nIMPROVEMENT:")
    for metric_name in base_metrics.keys():
        base_val = base_metrics[metric_name]
        finetuned_val = finetuned_metrics[metric_name]
        improvement = ((finetuned_val - base_val) / base_val * 100) if base_val > 0 else 0
        sign = "+" if improvement > 0 else ""
        print(f"  {metric_name.upper():<20} {sign}{improvement:.2f}%")
    
    print("=" * 50)
    
    print("\n[Validation]")
    if finetuned_metrics["recall@10"] > base_metrics["recall@10"]:
        print("Fine-tuned model shows improvement in Recall@10")
    else:
        print("Fine-tuned model did not improve Recall@10")
    
    if finetuned_metrics["mrr@10"] > base_metrics["mrr@10"]:
        print("Fine-tuned model shows improvement in MRR@10")
    else:
        print("Fine-tuned model did not improve MRR@10")
    
    if finetuned_metrics["ndcg@10"] > base_metrics["ndcg@10"]:
        print("Fine-tuned model shows improvement in NDCG@10")
    else:
        print("Fine-tuned model did not improve NDCG@10")
    
    print()


if __name__ == "__main__":
    main()
