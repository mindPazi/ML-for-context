from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TrainingConfig:
    base_model_name: str = "microsoft/unixcoder-base"
    output_model_path: str = "./models/unixcoder-finetuned"
    max_seq_length: int = 256
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    use_amp: bool = True
    random_seed: int = 42
    train_split: float = 0.8
    val_split: float = 0.2
    num_workers: int = 4
    patience: int = 3
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.use_amp = False
                self.num_workers = 0  
            else:
                self.device = "cpu"
                self.use_amp = False
                self.num_workers = 0  
