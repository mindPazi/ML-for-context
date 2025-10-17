# ML for Context - Embeddings-based Code Search Engine

A semantic code search engine using embedding models, with support for fine-tuning on the CoSQA dataset.

## Features

- **Embeddings-based Search**: Uses transformer models to create semantic embeddings of code and queries
- **Vector Search**: Fast similarity search using cosine similarity with normalized embeddings
- **Model Flexibility**: Support for both base models and fine-tuned models
- **Fine-tuning Pipeline**: Complete infrastructure for fine-tuning models on CoSQA dataset
- **Comprehensive Evaluation**: Recall@10, MRR@10, and NDCG@10 metrics on CoSQA benchmark

## Project Structure

```
ML-for-context/
├── README.md
├── requirements.txt
├── src/                           # Core search engine (reusable)
│   ├── __init__.py
│   ├── embeddings.py             # Embedding model wrapper
│   ├── search_engine.py          # Search engine with vector retrieval
│   ├── vector_store.py           # In-memory vector storage
│   └── api.py                    # FastAPI endpoints
├── evaluation/                    # Evaluation pipeline
│   ├── cosqa_loader.py           # CoSQA dataset loader
│   ├── metrics.py                # Ranking metrics (Recall, MRR, NDCG)
│   └── evaluate.py               # Evaluation script
├── training/                      # Fine-tuning infrastructure
│   ├── __init__.py
│   ├── config.py                 # Training configuration
│   ├── data_preparation.py       # Data preparation for fine-tuning
│   └── train.py                  # Fine-tuning script
├── tests/
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│       ├── test_evaluate_subset.py
│       └── test_evaluate_finetuned.py
└── models/                        # Fine-tuned models (gitignored)
```

## Requirements

- Python 3.8+
- PyTorch (with MPS support for Mac GPU acceleration)
- sentence-transformers
- datasets (for CoSQA)
- numpy
- FastAPI (optional, for API server)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Evaluate Base Model

Evaluate the pre-trained UniXcoder model on CoSQA test set:

```bash
python -m evaluation.evaluate --model microsoft/unixcoder-base
```

Expected results (base model):
- Recall@10: ~0.29 (29% of queries find relevant doc in top 10)
- MRR@10: ~0.13
- NDCG@10: ~0.17

### 2. Fine-tune Model

Fine-tune the model on CoSQA training data to improve performance:

```bash
python -m training.train
```

Optional arguments:
```bash
python -m training.train \
  --base-model microsoft/unixcoder-base \
  --output-path ./models/unixcoder-finetuned \
  --batch-size 16 \
  --epochs 3 \
  --learning-rate 2e-5
```

The fine-tuning process:
- Loads full CoSQA dataset
- Creates query-document pairs from relevance judgments
- Splits into 80% train / 20% validation
- Uses MultipleNegativesRankingLoss (contrastive learning)
- Saves model to `./models/unixcoder-finetuned/`

### 3. Evaluate Fine-tuned Model

Compare fine-tuned model against base model:

```bash
# Evaluate fine-tuned model
python -m evaluation.evaluate --model ./models/unixcoder-finetuned

# Or run comparison test
python -m tests.integration.test_evaluate_finetuned
```

Expected improvement:
- Recall@10: 0.29 → 0.50-0.65 (50-65% of queries successful)
- Significant improvements in MRR and NDCG as well

### 4. Run Integration Tests

```bash
# Test on subset (quick validation)
python -m tests.integration.test_evaluate_subset

# Test fine-tuned vs base model comparison
python -m tests.integration.test_evaluate_finetuned
```

## Model Architecture

### Base Model: microsoft/unixcoder-base
- Transformer encoder specifically designed for code understanding
- Pre-trained on 1M+ code functions from GitHub
- 125M parameters
- Max sequence length: 256 tokens
- Embedding dimension: 768

### Fine-tuning Strategy
- **Loss Function**: MultipleNegativesRankingLoss
  - Uses in-batch negatives for contrastive learning
  - Given batch of (query, positive_doc) pairs, treats other docs as negatives
  - Learns to pull queries closer to relevant docs, push away from irrelevant ones
- **Dataset**: CoSQA (20,604 code snippets, 20,604 queries)
- **Training Split**: 80% train, 20% validation
- **Device**: Automatic detection (CUDA > MPS > CPU)

## Evaluation Metrics

### Recall@10
Percentage of queries where at least one relevant document appears in top 10 results.

### MRR@10 (Mean Reciprocal Rank)
Average of reciprocal ranks of first relevant document. Rewards relevant docs appearing earlier.

### NDCG@10 (Normalized Discounted Cumulative Gain)
Accounts for position and number of relevant documents. Logarithmic discount for lower positions.

## Bonus Experiments & Analysis

The `report.ipynb` notebook contains comprehensive experiments and analysis:

### Bonus 1: Function Names Only
Evaluates search performance using only extracted function names instead of full code bodies:
- Recall@10: 0.1560 (vs 0.4260 with full code)
- Demonstrates importance of full code context

### Bonus 2: Similarity Metrics Comparison
Compares different distance metrics with normalized/unnormalized embeddings:
- **Best**: Manhattan distance with normalized embeddings (Recall@10: 0.4260)
- Cosine similarity: 0.3860
- Euclidean distance: 0.3860  
- Dot product: 0.3860

### Embeddings Analysis - Anisotropy Detection
Analysis of fine-tuned embedding structure:
- Strong anisotropy detected: only 31 dimensions explain 50% of variance
- 83.7% of embedding values are near-zero (< 0.05)
- Embeddings concentrated in low-dimensional subspace

### PCA and PCA Whitening Analysis
Evaluates dimensionality reduction and decorrelation techniques:
- **PCA (768 → 256 dims)**: Minor performance loss (-2-3%)
- **PCA Whitening**: Significant performance degradation (-14-25%)
- Conclusion: Anisotropic structure is beneficial for code search

Run the full analysis notebook:
```bash
jupyter notebook report.ipynb
```

Additional experiment scripts:
```bash
# Bonus 1: Function names vs whole bodies
python src/bonus/bonus1.py

# Bonus 2: Vector storage metrics comparison
python src/bonus/bonus2.py

# Compare all results
python src/bonus/compare.py

# Analyze embeddings for anisotropy
python src/bonus/analyze_embeddings.py
```

## API Usage (Optional)

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

Search endpoint:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "function to read json file", "top_k": 10}'
```

## Development

### Project Design Principles

1. **No Code Duplication**: Core modules (embeddings, search_engine, vector_store) are reusable and model-agnostic
2. **Model Flexibility**: Pass `model_name` parameter to switch between base and fine-tuned models
3. **Clean Separation**: 
   - `src/`: Core reusable components
   - `evaluation/`: Evaluation pipeline
   - `training/`: Fine-tuning infrastructure
   - `tests/`: Validation and testing

### Key Implementation Details

#### Index Mapping Fix
The original implementation had a critical bug using `corpus.index(text)` which fails with duplicate documents. Fixed by returning integer indices directly from search engine:

```python
# search_engine.py - returns integer index
result = {
    "id": int(idx),  # Integer index into corpus
    "text": self.vector_store.documents[idx],
    "score": float(similarities[idx])
}
```

#### Model Selection
Changed from CodeBERT to UniXcoder because:
- UniXcoder specifically designed for code search tasks
- Better understanding of code syntax and semantics
- Pre-trained on larger and more diverse code corpus

## Citation

CoSQA Dataset:
```
@article{huang2021cosqa,
  title={CoSQA: 20,000+ Web Queries for Code Search and Question Answering},
  author={Huang, Junjie and Tang, Duyu and Shou, Linjun and Gong, Ming and Xu, Ke and Jiang, Daxin and Zhou, Ming and Duan, Nan},
  journal={arXiv preprint arXiv:2105.13239},
  year={2021}
}
```

## License

MIT License
