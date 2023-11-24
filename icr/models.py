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
from icr.aux import mask_pads

from icr.components import (
    ActionsMaker, CrossEncoder, iCRClipDecoder, iCRTurnDecoder,
    SceneEncoder, SelfCrossEncoder, StateEmbedding, iCRDecoder)


class TopicModel(nn.Module):
    
    def __init__(self, model_config = Dict[str, int]):
        super().__init__()

        self.icr_clip_decoder = iCRClipDecoder(
                d_model=model_config["d_model"], hidden_dim=model_config["hidden_dim"], dropout=model_config["dropout"])
        self.labels = self.icr_clip_decoder.labels
    
    def forward(self, inputs: Dict[str, Tensor],
                labels: Dict[str, Tensor]) -> Dict[str, Tensor]:

        icr_outputs = {}
        icr_outputs = self.icr_clip_decoder(inputs)

        return {**icr_outputs}

    
class ICRModel(nn.Module):
    
    def __init__(self, model_config = Dict[str, int]):
        super().__init__()

        self.icr_decoder = iCRDecoder(
                d_model=model_config["d_model"], hidden_dim=model_config["hidden_dim"], 
                output_dim = model_config["nlayers"], nlayers = model_config["d_model"], dropout=model_config["dropout"])
        self.labels = self.icr_decoder.labels
    
    def forward(self, inputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Dict[str, Tensor]:
        icr_outputs = {}
        icr_outputs = self.icr_decoder(inputs)

        return {**icr_outputs}
    

    
class TextEncoder(nn.Module):
    """Encoding tokenized text input"""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        
        self.seqencoder1 = nn.Embedding(d_model, hidden_dim, padding_idx=0)
        self.seqencoder2 = nn.Dropout(p=dropout)
        self.seqencoder3 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, inputs: Tensor) -> Tensor:
        
        se1_out = F.relu(self.seqencoder1(inputs))
        se2_out = self.seqencoder2(se1_out)
        se3_out = self.seqencoder3(se2_out)
        return se3_out
    
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
        
        self.decoder1 = nn.LSTM(d_model, hidden_dim, num_layers, dropout=dropout, batch_first=True, bias=True)
        self.decoder2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        
        decoder1_out, _ = self.decoder1(inputs)
        outputs = self.decoder2(self.dropout(decoder1_out))
        #decoder2_out = self.decoder2(F.relu(decoder1_out[:,-1,:)
        #outputs = F.softmax(decoder2_out, dim = 1)
        
        return outputs  
    
