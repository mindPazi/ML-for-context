from src.search_engine import SearchEngine
from evaluation.cosqa_loader import CoSQALoader
from evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
log = logging.getLogger(__name__)


def load_dataset_subset(split: str = "test", max_queries: int = 10):
    loader = CoSQALoader()
    corpus, queries, relevance = loader.load(split=split)
    
    queries_subset = queries[:max_queries]
    
    doc_indices = set()
    for query in queries_subset:
        query_id = query["query_id"]
        if query_id in relevance:
            doc_indices.update(relevance[query_id])
    
    doc_indices = sorted(list(doc_indices))
    
    corpus_subset = [corpus[idx] for idx in doc_indices]
    
    idx_to_corpus_pos = {orig_idx: pos for pos, orig_idx in enumerate(doc_indices)}
    
    relevance_subset = {}
    for query in queries_subset:
        query_id = query["query_id"]
        if query_id in relevance:
            corpus_positions = [idx_to_corpus_pos[orig_idx] for orig_idx in relevance[query_id] if orig_idx in idx_to_corpus_pos]
            if corpus_positions:
                relevance_subset[query_id] = corpus_positions
    
    return corpus_subset, queries_subset, relevance_subset


def index_corpus(search_engine: SearchEngine, corpus: List[str]):
    search_engine.index_documents(corpus, show_progress=True)


def run_queries(search_engine: SearchEngine, queries: List[Dict], corpus: List[str], top_k: int = 10):
    results = {}
    for query in queries:
        query_id = query["query_id"]
        query_text = query["query_text"]
        search_results = search_engine.search(query_text, top_k=top_k)
        retrieved_indices = [corpus.index(r["text"]) for r in search_results]
        results[query_id] = retrieved_indices
    return results


def calculate_metrics(results: Dict, relevance: Dict):
    recalls = []
    mrrs = []
    ndcgs = []
    
    for query_id, retrieved in results.items():
        if query_id in relevance:
            relevant = relevance[query_id]
            recalls.append(recall_at_k(relevant, retrieved, k=10))
            mrrs.append(mrr_at_k(relevant, retrieved, k=10))
            ndcgs.append(ndcg_at_k(relevant, retrieved, k=10))
    
    if not recalls:
        return {"recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}
    
    metrics = {
        "recall@10": sum(recalls) / len(recalls),
        "mrr@10": sum(mrrs) / len(mrrs),
        "ndcg@10": sum(ndcgs) / len(ndcgs)
    }
    return metrics


def print_results(metrics: Dict):
    log.info("\n")
    log.info("EVALUATION RESULTS (SUBSET)".center(50))
    log.info("\n")
    for metric_name, value in metrics.items():
        log.info(f"  {metric_name.upper():<20} {value:.4f}")
    log.info("\n")


def main():
    log.info("\n")
    log.info("CoSQA EVALUATION - SUBSET TEST".center(50))
    log.info("\n")
    
    log.info("\n[1/5] Loading dataset subset...")
    corpus, queries, relevance = load_dataset_subset(split="test", max_queries=10)
    log.info(f"      ├─ Corpus size: {len(corpus)}")
    log.info(f"      └─ Queries: {len(queries)}")
    
    log.info("\n[2/5] Initializing search engine...")
    search_engine = SearchEngine()
    
    log.info("\n[3/5] Indexing corpus...")
    index_corpus(search_engine, corpus)
    
    log.info("\n[4/5] Running queries...")
    results = run_queries(search_engine, queries, corpus, top_k=10)
    
    log.info("\n[5/5] Calculating metrics...")
    metrics = calculate_metrics(results, relevance)
    
    print_results(metrics)


if __name__ == "__main__":
    main()
