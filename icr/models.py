#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from typing import Dict, List, Tuple

from positional_encodings.torch_encodings import PositionalEncoding1D
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from icr import constants

    
class ProbDecoder(nn.Module):
    """A classifier for the probability of one event (as logit)."""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.propdecoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.propdecoder(inputs)
    
    
class TextEncoder(nn.Module):
    """Encoding tokenized text input"""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        
        self.emb = nn.Embedding(d_model, hidden_dim, padding_idx=0)
        self.fc = nn.Linear(hidden_dim, output_dim) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        
        x = F.relu(self.emb(inputs))
        x = self.dropout(x)
        ouput = self.fc(x)
        return ouput
    
    
class EmbeddingCompresser(nn.Module):
    """Compressing loaded text embeddings"""    
    def __init__(self, output_dim):
        super().__init__()
        
        self.compresser = nn.Linear(320*768, output_dim)
        
    def forward(self, inputs: Tensor) -> Tensor:
        
        return self.compresser(torch.flatten(inputs,1,2))

    
class TextDecoder(nn.Module):
    """Decoding into probabilities over the vocabulary"""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(d_model, hidden_dim, num_layers, dropout=dropout, batch_first=True, bias=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: Tensor, features: Tensor) -> Tensor:
        
        h0 = features.repeat(self.num_layers, 1, 1)  #(num_layers, batch_size, hidden_size)
        c0 = torch.zeros_like(h0)
        
        decoder1_out, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.fc(self.dropout(decoder1_out))
        
        return outputs  
    
