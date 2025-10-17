import pickle
import numpy as np
from sklearn.decomposition import PCA
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.embeddings import EmbeddingModel
from src.evaluation.cosqa_loader import CoSQALoader

print("Loading test set...")
loader = CoSQALoader()
corpus, queries, relevance = loader.load(split='test')
print(f"Loaded {len(corpus)} documents and {len(queries)} queries")

print("\nLoading cached embeddings...")
with open('./cache/embeddings/embeddings_finetuned_normalized.pkl.npz', 'rb') as f:
    embeddings = pickle.load(f)

print("\nComputing PCA...")
pca = PCA()
pca_embeddings = pca.fit_transform(embeddings)
pca_embeddings = pca_embeddings / np.linalg.norm(pca_embeddings, axis=1, keepdims=True)

print("Computing PCA whitening...")
pca_whitened = PCA(whiten=True)
pca_whitened_embeddings = pca_whitened.fit_transform(embeddings)
pca_whitened_embeddings = pca_whitened_embeddings / np.linalg.norm(pca_whitened_embeddings, axis=1, keepdims=True)

print("\nLoading model for query encoding...")
model = EmbeddingModel(
    model_name="./models/unixcoder-finetuned",
    max_seq_length=256,
    device=None
)

def process_queries_and_compute_metrics(queries, corpus_embeddings, query_transform_fn, metric_type='cosine'):
    recalls = []
    mrrs = []
    ndcgs = []
    
    for query in queries:
        query_id = query["query_id"]
        query_text = query["query_text"]
        relevant_indices = relevance[query_id]
        
        query_embedding = model.encode(query_text, batch_size=1, show_progress_bar=False)
        
        if query_transform_fn is not None:
            query_embedding = query_transform_fn(query_embedding.reshape(1, -1))
            query_embedding = query_embedding.flatten()
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        else:
            query_embedding = query_embedding.flatten()
        
        if metric_type == 'cosine':
            scores = np.dot(corpus_embeddings, query_embedding)
            top_indices = np.argsort(scores)[-10:][::-1]
        else:
            distances = np.sum(np.abs(corpus_embeddings - query_embedding), axis=1)
            top_indices = np.argsort(distances)[:10]
        
        retrieved = top_indices.tolist()
        relevant_set = set(relevant_indices)
        retrieved_set = set(retrieved[:10])
        
        recalls.append(len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0)
        
        for i, doc_idx in enumerate(retrieved[:10], 1):
            if doc_idx in relevant_set:
                mrrs.append(1.0 / i)
                break
        else:
            mrrs.append(0.0)
        
        dcg = 0
        for i, doc_idx in enumerate(retrieved[:10], 1):
            if doc_idx in relevant_set:
                dcg += 1.0 / np.log2(i + 1)
        
        idcg = 0
        for i in range(1, min(len(relevant_set), 10) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    return (
        sum(mrrs) / len(mrrs),
        sum(recalls) / len(recalls),
        sum(ndcgs) / len(ndcgs)
    )

print("\nComputing baseline metrics...")
baseline_cosine = process_queries_and_compute_metrics(queries, embeddings, None, 'cosine')
baseline_manhattan = process_queries_and_compute_metrics(queries, embeddings, None, 'manhattan')

print("Computing PCA metrics...")
pca_cosine = process_queries_and_compute_metrics(queries, pca_embeddings, pca.transform, 'cosine')
pca_manhattan = process_queries_and_compute_metrics(queries, pca_embeddings, pca.transform, 'manhattan')

print("Computing whitened metrics...")
whitened_cosine = process_queries_and_compute_metrics(queries, pca_whitened_embeddings, pca_whitened.transform, 'cosine')
whitened_manhattan = process_queries_and_compute_metrics(queries, pca_whitened_embeddings, pca_whitened.transform, 'manhattan')

results = {
    'baseline': {
        'cosine': baseline_cosine,
        'manhattan': baseline_manhattan
    },
    'pca': {
        'cosine': pca_cosine,
        'manhattan': pca_manhattan
    },
    'whitened': {
        'cosine': whitened_cosine,
        'manhattan': whitened_manhattan
    }
}

print("\n\nRESULTS ON TEST SET:")
print("="*60)
for method in ['baseline', 'pca', 'whitened']:
    print(f"\n{method.upper()}:")
    for metric in ['cosine', 'manhattan']:
        mrr, recall, ndcg = results[method][metric]
        print(f"  {metric}:")
        print(f"    MRR@10:    {mrr:.4f}")
        print(f"    Recall@10: {recall:.4f}")
        print(f"    NDCG@10:   {ndcg:.4f}")
