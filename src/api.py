from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from .search_engine import SearchEngine
import uvicorn
app = FastAPI()
engine = SearchEngine()


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
