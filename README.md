# ML for Context - Embeddings-based Code Search Engine

## Task Overview

### Part 1: Embeddings-based Search Engine
- Accept a collection of documents on start
- Provide an API to search over this collection by text query
- Use pretrained embeddings model from Hugging Face for vector representations
- Vector storage and retrieval using usearch, Weaviate, or Qdrant
- Demonstrate on test samples from any source

### Part 2: Evaluation
- Apply search engine to code search task
- Evaluate on CoSQA dataset
- Implement ranking metrics: Recall@10, MRR@10, and NDCG@10

## Project Structure

```
ML-for-context/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── search_engine.py
│   ├── vector_store.py
│   └── api.py
├── evaluation/
│   ├── __init__.py
│   ├── cosqa_loader.py
│   ├── metrics.py
│   └── evaluate.py
├── data/
│   └── sample_docs/
├── tests/
│   └── test_search.py
└── main.py
```

## Requirements

- Python 3.8+
- Hugging Face Transformers
- Vector database (one of):
  - usearch
  - Weaviate
  - Qdrant
- FastAPI for API endpoints
- NumPy for calculations
- Pandas for data handling

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Basic Search Engine

```python
python main.py
```

### Evaluation on CoSQA

```python
python evaluation/evaluate.py
```

## Implementation Status

- [ ] Part 1: Search Engine
  - [ ] Document loading
  - [ ] Embeddings generation
  - [ ] Vector storage
  - [ ] Search API
  - [ ] Demo with test samples
- [ ] Part 2: Evaluation
  - [ ] CoSQA dataset loading
  - [ ] Recall@10 metric
  - [ ] MRR@10 metric
  - [ ] NDCG@10 metric
  - [ ] Full evaluation pipeline
