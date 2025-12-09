# file 1: config.py
"""Configuration for CLINC150 experiments."""

from dataclasses import dataclass
import torch

@dataclass
class Config:
    # Model
    model_name: str = 'distilbert-base-uncased'
    num_classes: int = 150
    dropout_rate: float = 0.1

    # Data
    max_seq_length: int = 64
    batch_size: int = 64

    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 8
    patience: int = 3
    min_improvement: float = 0.002

    # Evaluation
    num_mc_samples: int = 50

    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed: int = 42