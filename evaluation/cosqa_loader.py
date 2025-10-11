from datasets import load_dataset
from typing import List, Dict, Tuple


class CoSQALoader:
    def load(self, split: str = "test") -> Tuple[List[str], List[Dict], Dict]:
        
        qrels = load_dataset("CoIR-Retrieval/cosqa", split=split)
        corpus_ds = load_dataset("CoIR-Retrieval/cosqa", "corpus")["corpus"]
        queries_ds = load_dataset("CoIR-Retrieval/cosqa", "queries")["queries"]
        
        
        corpus_dict = {}
        for item in corpus_ds:
            doc_id = item["_id"]
            doc_text = item["text"]
            corpus_dict[doc_id] = doc_text
        
        
        queries_dict = {}
        for item in queries_ds:
            query_id = item["_id"]
            query_text = item["text"]
            queries_dict[query_id] = query_text
        
        
        corpus = []
        for doc_text in corpus_dict.values():
            corpus.append(doc_text)
        
        
        id_to_idx = {}
        idx = 0
        for doc_id in corpus_dict.keys():
            id_to_idx[doc_id] = idx
            idx += 1
        
        
        queries = []
        relevance = {}
        seen = set()
        
        for row in qrels:
            q_id = row["query-id"]
            doc_id = row["corpus-id"]
            
            
            if q_id not in seen:
                query_obj = {
                    "query_id": q_id,
                    "query_text": queries_dict[q_id]
                }
                queries.append(query_obj)
                relevance[q_id] = []
                seen.add(q_id)
            
            
            doc_idx = id_to_idx[doc_id]
            relevance[q_id].append(doc_idx)
        
        return corpus, queries, relevance
