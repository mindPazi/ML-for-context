import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search_engine import SearchEngine
from evaluation.cosqa_loader import CoSQALoader
from evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k
import json
from typing import Dict, List
import numpy as np


def evaluate_with_metrics():
    print("\n" + "="*60)
    print("BONUS 2: Vector Storage Hyperparameters")
    print("="*60)
    
    # TODO: Test different similarity metrics:
    # TODO: - Cosine (baseline)
    # TODO: - Euclidean distance
    # TODO: - Manhattan distance
    # TODO: - Dot product (no normalization)
    # TODO: Save results to bonus/results/metrics_comparison.json
    
    print("\nTODO: Implement metrics comparison")


if __name__ == "__main__":
    evaluate_with_metrics()
