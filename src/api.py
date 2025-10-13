from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from .search_engine import SearchEngine
from training.config import TrainingConfig
import uvicorn

app = FastAPI()

config = TrainingConfig()
engine = SearchEngine(
    model_name=config.base_model_name,
    max_seq_length=config.max_seq_length,
    device=None
)


class IndexRequest(BaseModel):
    documents: List[str]
    metadata: Optional[List[Dict]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/index")
def index_documents(request: IndexRequest):
    count = engine.index_documents(request.documents, request.metadata)
    return {"indexed": count}


@app.post("/search")
def search(request: SearchRequest):
    results = engine.search(request.query, request.top_k)
    return {"results": results}


@app.get("/info")
def get_info():
    return engine.info()


@app.delete("/clear")
def clear_collection():
    engine.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
