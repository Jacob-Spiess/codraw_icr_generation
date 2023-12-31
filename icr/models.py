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

    
class SingleProbDecoder(nn.Module):
    """A classifier for the probability of one event (as logit)."""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.propdecoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.propdecoder(inputs)

class MultiProbDecoder(nn.Module):
    """A classifier for the probability of multiple event."""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.propdecoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())

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
    
    
class EmbeddingExtender(nn.Module):
    """Linear layer to map Embeddings to a dimension"""    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.extender = nn.Linear(input_dim, output_dim)
        
    def forward(self, inputs: Tensor) -> Tensor:
        
        return self.extender(inputs)
    
    
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
    
    
class Attention(nn.Module):
    """
    Attention Network modified for encoder output of shape (batch_size, encoder_dim).
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)


    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)                    # (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)                 # (batch_size, attention_dim)
        combined = torch.tanh(att1 + att2)                      # (batch_size, attention_dim)
        att = self.full_att(combined)                           # (batch_size, 1)
        alpha = F.softmax(att, dim=1)                           # (batch_size, 1)
        attention_weighted_encoding = encoder_out * alpha       # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha.squeeze(1)
    
    
class AttentionTextDecoder(nn.Module):
    def __init__(self, embedding_size, encoder_dim, decoder_dim, attention_dim, vocab_size, vocab, drop_prob = 0.2):
        super().__init__()
        
        self.vocab = vocab
        
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embedding_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid() 
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
    
    def forward(self, embeddings, features):     

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)                              # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = embeddings.size(1) 
        batch_size = embeddings.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size)
        alphas = torch.zeros(batch_size, seq_length)
                
        for s in range(seq_length):
            attention_weighted_encoding, alpha = self.attention(features, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            lstm_input = torch.cat((embeddings[:, s], attention_weighted_encoding), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            
            preds[:, s] = output
            alphas[:, s] = alpha  
        
        return preds, alphas
    
    def init_hidden_state(self, encoder_out):
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c
    

class TransformerDecoder(nn.Module):
    """Implements an encoder with a memory."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, nheads: int, nlayers: int, dropout: float):
        super().__init__()
        trf_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nheads, batch_first=True, 
                                               dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer = nn.TransformerDecoder(trf_layer, num_layers=nlayers)
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, target: Tensor, memory: Tensor) -> Tensor:
        target_len = target.size(1)
        causal_mask = self.generate_square_subsequent_mask(target_len)

        out = self.transformer(target, memory, tgt_mask=causal_mask)
        outputs = self.fc(self.dropout(out))
        return outputs  
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    