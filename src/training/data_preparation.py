import random
from typing import List, Tuple, Dict
from sentence_transformers import InputExample

from src.evaluation.cosqa_loader import CoSQALoader
from src.training.config import TrainingConfig


def prepare_data(config: TrainingConfig) -> Tuple[List[InputExample], List[str], List[Dict], Dict]:
    print("\n[Data Preparation]")
    print("=" * 50)
    
    print("[1/3] Loading CoSQA dataset...")
    loader = CoSQALoader()
    corpus, queries, qrels = loader.load(split="train")
    print(f"      Corpus size: {len(corpus)}")
    print(f"      Total queries: {len(queries)}")
    
    print("\n[2/3] Splitting queries into train/validation (80/20)...")
    random.seed(config.random_seed)
    random.shuffle(queries)
    
    val_split_idx = int(len(queries) * config.train_split)
    train_queries = queries[:val_split_idx]
    val_queries = queries[val_split_idx:]
    
    print(f"      Train queries: {len(train_queries)} ({config.train_split*100:.0f}%)")
    print(f"      Val queries: {len(val_queries)} ({config.val_split*100:.0f}%)")
    print(f"      (Test queries: 500 from CoSQA test split)")
    
    print("\n[3/3] Creating query-document pairs for training...")
    train_pairs = []
    for query in train_queries:
        query_id = query["query_id"]
        query_text = query["query_text"]
        
        if query_id in qrels:
            relevant_doc_indices = qrels[query_id]
            for doc_idx in relevant_doc_indices:
                if doc_idx >= len(corpus):
                    raise IndexError(
                        f"Document index {doc_idx} out of bounds for corpus of size {len(corpus)} "
                        f"(query_id: {query_id})"
                    )
                doc_text = corpus[doc_idx]
                train_pairs.append((query_text, doc_text))
    
    print(f"      Train pairs: {len(train_pairs)}")
    
    train_examples = [InputExample(texts=[query, doc]) for query, doc in train_pairs]
    
    print("=" * 50)
    
    return train_examples, corpus, val_queries, qrels
