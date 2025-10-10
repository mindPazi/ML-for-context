from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Union


class EmbeddingModel:
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32, #to avoid out of memory
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True #to make cosine similarity a dot similarity for fast search
    ) -> np.ndarray:
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
