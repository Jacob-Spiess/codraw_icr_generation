#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All models for encoding and decoding that are utilized in the iCR generation model. 
"""
from typing import Dict, List, Tuple

from positional_encodings.torch_encodings import PositionalEncoding1D
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import BartConfig, BartForCausalLM

from icr import constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    def forward(self, inputs: Tensor) -> Tensor:            # Input: (batch_size, input_dim)
        return self.propdecoder(inputs)                     # Output: (batch_size, output_dim)

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
            nn.Linear(hidden_dim, output_dim)#,
            #nn.Sigmoid()
        )

    def forward(self, inputs: Tensor) -> Tensor:            # Input: (batch_size, input_dim)
        return self.propdecoder(inputs)                     # Output: (batch_size, output_dim)
    
class TextEncoder(nn.Module):
    """Encoding tokenized text input utilizing an embedding layer"""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, dropout: float, pad = 0):
        super().__init__()
        
        self.emb = nn.Embedding(d_model, hidden_dim, padding_idx=pad)
        self.fc = nn.Linear(hidden_dim, output_dim) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor) -> Tensor:            # Input: (batch_size)
        x = F.relu(self.emb(inputs))
        x = self.dropout(x)
        ouput = self.fc(x)
        return ouput                                        # Output: (batch_size, output_dim)
    
    
class EmbeddingExtender(nn.Module):
    """
    Linear layer to map Embeddings to a dimension. Used to increase the dimensionality of the classifier outputs to match the decoder size
    """    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.extender = nn.Linear(input_dim, output_dim)
        
    def forward(self, inputs: Tensor) -> Tensor:             # Input: (batch_size, input_dim)
        return self.extender(inputs)                         # Output: (batch_size, output_dim)
    
    
class EmbeddingCompresser(nn.Module):
    """Altering the text embedding size of the given dialogue data"""    
    def __init__(self, output_dim):
        super().__init__()
        
        self.compresser = nn.Linear(768, output_dim)                 
        
    def forward(self, inputs: Tensor) -> Tensor:
        """The input tensor come from the saved dialogue embeddings"""   # Input: (batch_size, sequence_length, 768)
        return self.compresser(inputs)                                   # Output: (batch_size, sequence_length, output_dim)
    
    
class EmbeddingCompresserLarge(nn.Module):
    """
    Compressing loaded text embeddings and removing the sequence_length dimension by flattening the input sequence 
    and processing it with a Linear Layer. Results in a model with many parameters that make the generation model 
    often to large for the available computation ressources.
    """    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.compresser = nn.Linear(input_dim, output_dim)
        
    def forward(self, inputs: Tensor) -> Tensor:                        # Input: (batch_size, sequence_length, input_dim)
        return self.compresser(torch.flatten(inputs,1,2))               # Output: (batch_size, output_dim)

    
class EmbeddingCompresserCNNScene(nn.Module): 
    """ CNN to reduce the complexity of the encoded scene data and turn it into a 2D Tensor, adjusted for varying input dimensions """
    def __init__(self, d_input, hidden_dim, output_dim):
        super(EmbeddingCompresserCNNScene, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(208, 128, kernel_size=3, stride=2, padding=1),  # Output: [32, 128, d_input/2**1]
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),   # Output: [32, 64, d_input/2**2]
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),    # Output: [32, 32, d_input/2**3]
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),    # Output: [32, 16, d_input/2**4]
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for the fully connected layers
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16*int(d_input/2**4), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):                                             # Input: (batch_size, sequence_length, input_dim)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x                                                      # Output: (batch_size, output_dim)

    
class EmbeddingCompresserCNNDialogue(nn.Module):
    """ CNN to reduce the complexity of the embedded dialogue data and turn it into a 2D Tensor for one fixed input dimension """
    def __init__(self, hidden_dim, output_dim):
        super(EmbeddingCompresserCNNDialogue, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(320, 160, kernel_size=3, stride=2, padding=1),  # Output: [32, 160, 384]
            nn.ReLU(),
            nn.Conv1d(160, 80, kernel_size=3, stride=2, padding=1),   # Output: [32, 80, 192]
            nn.ReLU(),
            nn.Conv1d(80, 40, kernel_size=3, stride=2, padding=1),    # Output: [32, 40, 96]
            nn.ReLU(),
            nn.Conv1d(40, 20, kernel_size=3, stride=2, padding=1),    # Output: [32, 20, 48]
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for the fully connected layers
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(20*48, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):                                            # Input: (batch_size, 320, 768) fixed!
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x                                                     # Output: (batch_size, output_dim)
    
    
    
class TextDecoder(nn.Module):
    """LSTM based implementation of a decoder with a memory outputing probabilities over the vocabulary"""
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.num_layers = num_layers
        self.lstm = nn.LSTM(d_model, hidden_dim, num_layers, dropout=dropout, batch_first=True, bias=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: Tensor, features: Tensor) -> Tensor: # Features: (batch_size, decoder_size)
        """Features are the memory fed into the model via the first hiddenstate"""
        h0 = features.repeat(self.num_layers, 1, 1)                    # h0: (num_layers, batch_size, decoder_size)
        c0 = torch.zeros_like(h0)
        
        out, _ = self.lstm(embeddings, (h0, c0))                       # Embedding: (batch_size, max_sequence_length, embedding_dim
        outputs = self.fc(self.dropout(out))                        
        return outputs                                                 # Output: (batch_size, max_sequence_length, output_dim)
    
    
class LearnablePosition(nn.Module):
    """Learns positional encoding for the scene feature embeddings.

    Originally from the DETR model https://arxiv.org/abs/2005.12872
    """
    def __init__(self, d_model: int, dropout: float, num_embeddings: int):
        super().__init__()
        self.row_embed = nn.Parameter(torch.rand(num_embeddings, d_model // 2))
        self.col_embed = nn.Parameter(torch.rand(num_embeddings, d_model // 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_scene: Tensor, height: int, width: int) -> Tensor:
        """Computes and adds positional embeddings to the input scene."""
        pos = torch.cat([
            self.col_embed[:width].unsqueeze(0).repeat(height, 1, 1),
            self.row_embed[:height].unsqueeze(1).repeat(1, width, 1)],
            dim=-1).flatten(0, 1).unsqueeze(1)
        return self.dropout(enc_scene + pos.permute(1, 0, 2))
    
class SceneEncoder(nn.Module):
    """Implements ResNet backbone for encoding scene, with CNN layer on top as done by (Madureira and Schlangen, 2024).

    Originally from the DETR model https://arxiv.org/abs/2005.12872
    Based on https://pytorch.org/vision/0.14/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
    """
    def __init__(self, d_model: int, dropout: float, unfreeze: bool, dont_preprocess_scenes: bool):
        super().__init__()
        self.preprocess_scenes = not dont_preprocess_scenes
        weights = ResNet18_Weights.DEFAULT  # Use ResNet18 weights
        if self.preprocess_scenes:
            self.img_preprocess = weights.transforms()
        model = list(resnet18(weights=weights).children())[:-2]
        self.backbone = nn.Sequential(*model)
        self.conv = nn.Conv2d(512, d_model, 1)  # Adjust the number of input channels based on the selected backbone
        self.positions = LearnablePosition(d_model, dropout, num_embeddings=50)
        if not unfreeze:
            self.freeze_params()

    def freeze_params(self) -> None:
        """Prevent fine-tuning of pretrained CV model."""
        for parameter in list(self.backbone.parameters()):
            parameter.requires_grad = False
        self.backbone.eval()

    def _preprocess(self, scene: Tensor) -> Tensor:
        if self.preprocess_scenes:
            return self.img_preprocess(scene)
        return scene.float() / constants.RGB_DIM

    def forward(self, scene: Tensor) -> Tensor:
        preproc_scene = self._preprocess(scene)
        features = self.conv(self.backbone(preproc_scene))
        height, width = features.shape[-2:]
        flattened = features.flatten(2).permute(0, 2, 1)
        return self.positions(flattened, height, width)
    
    
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
        self.transformer = nn.TransformerDecoder(trf_layer, num_layers=nlayers)#, tgt_is_causal=True)
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, target: Tensor, memory: Tensor, padding_mask: Tensor = None) -> Tensor: 
        #generate casual and padding mask                                     # Target: (batch_size, max_sequence_length, embedding_dim)
        target_len = target.size(1)                                           # Memory: (batch_size, sequence_length, decoder_dim)
        causal_mask = self.generate_square_subsequent_mask(target_len).to(device)
        if padding_mask is None:
            padding_mask = torch.zeros(target.shape[0], target.shape[1], dtype=torch.bool).to(device)
        
        #forward pass
        out = self.transformer(target, memory, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask)
        outputs = self.fc(self.dropout(out))                  
        return outputs                                                         # Output: (batch_size, max_sequence_length, output_dim)
    
    def generate_square_subsequent_mask(self, sz: int):
        """Generates a casual triangle mask, which is aligned to the right."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    

class BartTransformerDecoder(nn.Module):
    """
    Implements an encoder with a memory that utilizes a pretrained BART model from huggingfaces. 
    For more information on the BART encoder see: https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForCausalLM
    """
    
    def __init__(self, vocab_size: int, d_model: int, hidden_dim: int, nheads: int, nlayers: int, dropout: float):
        super().__init__()
        
        #Bart config
        self.bart_config = BartConfig.from_pretrained('facebook/bart-base', output_hidden_states=True)
        self.bart_config.add_cross_attention = True
        #self.bart_config.vocab_size = vocab_size
        self.bart_config.decoder_layers = nlayers
        self.bart_config.decoder_attention_heads = nheads
        self.bart_config.max_position_embeddings = 62
        self.bart_config.d_model = d_model
        self.bart_config.decoder_ffn_dim = hidden_dim
        
        #model architecture
        self.bart = BartForCausalLM(self.bart_config)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: Tensor, memory: Tensor) -> Tensor:              # Target: (batch_size, max_sequence_length, embedding_dim)
        """ Encoder gets memory as hiddenstates and target as input """       # Memory: (batch_size, sequence_length, decoder_dim)
        transformer_outputs = self.bart(encoder_hidden_states=memory, inputs_embeds= target)  
        outputs = self.fc(self.dropout(transformer_outputs.hidden_states[1]))
        return outputs                                                        # Output: (batch_size, max_sequence_length, output_dim)
