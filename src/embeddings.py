from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Union, Optional


class EmbeddingModel:
    def __init__(
        self, 
        model_name: str,
        max_seq_length: int,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.max_seq_length = max_seq_length
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32, 
        show_progress_bar: bool = False
    ) -> np.ndarray:
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
