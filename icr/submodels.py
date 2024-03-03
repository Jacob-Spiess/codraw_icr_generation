#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of PytorchLightining models that are saved and loaded into the full model framework later to ensure that the Classifier are trained as best as possible.
"""

from argparse import Namespace
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn, Tensor, optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F


from icr.aux import (mask_pads, split_batch, compute_accuracy_text, compute_accuracy_single_label, 
                     compute_accuracy_multi_label, load_partial_state_dict)
from icr.constants import REDUCTION, SPLITS, BOS, EOS, PAD, NUM_WEIGHTS, CLIP_WEIGHTS, TOPIC_WEIGHTS, MOOD_WEIGHTS
from icr.evaluator import Metrics, Outputs
from icr import constants
from icr.models import (TextEncoder, TextDecoder, EmbeddingCompresser, EmbeddingCompresserLarge, SingleProbDecoder, MultiProbDecoder, 
                        Attention, AttentionTextDecoder, TransformerDecoder, EmbeddingExtender, SceneEncoder, 
                        EmbeddingCompresserCNNScene, EmbeddingCompresserCNNDialogue)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NModel(pl.LightningModule):
    def __init__(self, vocab, config):
        super(NModel, self).__init__()

        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_scenes = config["use_scenes"]
        self.use_instructions = config["use_instructions"]

        self.loss_cross_n = CrossEntropyLoss(weight=torch.tensor(NUM_WEIGHTS), reduction=REDUCTION)

        num_clip_decoder_size = (
            config["decoder_size"] if self.use_instructions and not self.use_scenes
            else config["decoder_size"] * 2 if not self.use_instructions and self.use_scenes
            else config["decoder_size"] * 3 if self.use_instructions and self.use_scenes
            else print("No input available"))

        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.dialogue_encoder = EmbeddingCompresser(config["decoder_size"])
        self.num_clip_decoder = SingleProbDecoder(num_clip_decoder_size, config["hidden_dim"], config["num_clip_classes"],drop)
        
        self.scene_compresser = EmbeddingCompresserCNNScene(config["decoder_size"], config["hidden_dim"], config["decoder_size"])
        self.dialogue_compresser = EmbeddingCompresserCNNDialogue(config["hidden_dim"], config["decoder_size"])

        self.save_hyperparameters()
        self.to(device)
        

    def forward(self, dialogue, scene_before, scene_after):
        features1D = []
        if self.use_instructions:
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            features1D.append(self.scene_compresser(scene_before_features)) 
            features1D.append(self.scene_compresser(scene_after_features))
        features = torch.cat(features1D, dim=1).to(device)
        n = self.num_clip_decoder(features)
        return n
    
    def training_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, _, num_clip, _ = split_batch(batch)
        
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        num_clip = num_clip.to(device)

        n = self(dialogue, scene_before, scene_after)
        
        loss_n = self.calculate_loss(n, num_clip)

        self.log_metric("train", "loss", loss_n)
        return loss_n 

    def validation_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, _, num_clip, _ = split_batch(batch)

        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        num_clip = num_clip.to(device)

        n = self(dialogue, scene_before, scene_after)
        
        loss_n = self.calculate_loss(n, num_clip)
        acc_n = self.calculate_acc(n, num_clip)

        self.log_metric("val", "acc", acc_n)
        self.log_metric("val", "loss", loss_n)
        return loss_n
    
    def test_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, _, num_clip, _ = split_batch(batch)
        
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        num_clip = num_clip.to(device)

        n = self(dialogue, scene_before, scene_after)
        
        loss_n = self.calculate_loss(n, num_clip)
        acc_n = self.calculate_acc(n, num_clip)

        self.log_metric("test", "acc", acc_n)
        self.log_metric("test", "loss", loss_n)
        return loss_n 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr) 
    
    def calculate_loss(self, n, num_clip):
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss_n = self.loss_cross_n(n, num_clip.argmax(dim=1)) + l1_lambda * l1_norm
        return loss_n
        
    def calculate_acc(self, n, num_clip):        
        acc_n = compute_accuracy_single_label(n, num_clip.argmax(dim=1)) 
        return acc_n
    
    def log_metric(self, step, metric, n):
        self.log(f'{step}_{metric}_n', n)
        return

    
class CModel(pl.LightningModule):
    def __init__(self, vocab, config):
        super(CModel, self).__init__()
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_scenes = config["use_scenes"]
        self.use_instructions = config["use_instructions"]

        self.loss_bce_c = BCEWithLogitsLoss(pos_weight=torch.tensor(CLIP_WEIGHTS), reduction=REDUCTION)
        
        clipart_decoder_size = (
            config["decoder_size"] if self.use_instructions and not self.use_scenes
            else config["decoder_size"] * 2 if not self.use_instructions and self.use_scenes
            else config["decoder_size"] * 3 if self.use_instructions and self.use_scenes
            else print("No input available"))        

        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.dialogue_encoder = EmbeddingCompresser(config["decoder_size"])
        self.clipart_decoder = MultiProbDecoder(clipart_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)

        self.scene_compresser = EmbeddingCompresserCNNScene(config["decoder_size"], config["hidden_dim"], config["decoder_size"])
        self.dialogue_compresser = EmbeddingCompresserCNNDialogue(config["hidden_dim"], config["decoder_size"])

        self.save_hyperparameters()
        self.to(device)
        

    def forward(self, dialogue, scene_before, scene_after):

        features1D = []
        if self.use_instructions:
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            features1D.append(self.scene_compresser(scene_before_features)) 
            features1D.append(self.scene_compresser(scene_after_features))

        features = torch.cat(features1D, dim=1).to(device)
 
        c = self.clipart_decoder(features)
        return c
    
    def training_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, clip, _, _, _ = split_batch(batch)
        
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)

        c = self(dialogue, scene_before, scene_after)
        
        loss_c = self.calculate_loss(c, clip)

        self.log_metric("train", "loss", loss_c)
        return loss_c

    def validation_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, clip, _, _, _ = split_batch(batch)

        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)

        c = self(dialogue, scene_before, scene_after)
        
        loss_c = self.calculate_loss(c, clip)
        acc_c = self.calculate_acc(c, clip)

        self.log_metric("val", "acc", acc_c)
        self.log_metric("val", "loss", loss_c)
        
        return loss_c  
    
    def test_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, clip, _, _, _ = split_batch(batch)
        
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)

        c = self(dialogue, scene_before, scene_after)
        
        loss_c = self.calculate_loss(c, clip)
        acc_c = self.calculate_acc(c, clip)

        self.log_metric("test", "acc", acc_c)
        self.log_metric("test", "loss", loss_c)
        return loss_c

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr) 
    
    def calculate_loss(self, c, clip):
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss_c = self.loss_bce_c(c, clip.float()) + l1_lambda * l1_norm
        return loss_c
        
    def calculate_acc(self, c, clip):        
        acc_c = compute_accuracy_multi_label(c, clip.float()) 
        return acc_c
    
    def log_metric(self, step, metric, c):
        self.log(f'{step}_{metric}_c', c)
        return
    
    
class TModel(pl.LightningModule):
    def __init__(self, vocab, config):
        super(TModel, self).__init__()
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_scenes = config["use_scenes"]
        self.use_instructions = config["use_instructions"]

        self.loss_bce_t = BCEWithLogitsLoss(pos_weight=torch.tensor(TOPIC_WEIGHTS), reduction=REDUCTION)
        
        topic_decoder_size = (
            config["decoder_size"] if self.use_instructions and not self.use_scenes
            else config["decoder_size"] * 2 if not self.use_instructions and self.use_scenes
            else config["decoder_size"] * 3 if self.use_instructions and self.use_scenes
            else print("No input available"))
        
        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.dialogue_encoder = EmbeddingCompresser(config["decoder_size"])
        self.topic_decoder = MultiProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
        
        self.scene_compresser = EmbeddingCompresserCNNScene(config["decoder_size"], config["hidden_dim"], config["decoder_size"])
        self.dialogue_compresser = EmbeddingCompresserCNNDialogue(config["hidden_dim"], config["decoder_size"])

        self.save_hyperparameters()
        self.to(device)
        

    def forward(self, dialogue, scene_before, scene_after):

        features1D = []
        if self.use_instructions:
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            features1D.append(self.scene_compresser(scene_before_features)) 
            features1D.append(self.scene_compresser(scene_after_features))

        features = torch.cat(features1D, dim=1).to(device)

        t = self.topic_decoder(features)
        return t
    
    def training_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, _, _, topic = split_batch(batch)

        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        topic = topic.to(device)

        t = self(dialogue, scene_before, scene_after)
        
        loss_t = self.calculate_loss(t, topic)

        self.log_metric("train", "loss", loss_t)
        return loss_t  

    def validation_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, _, _, topic = split_batch(batch)

        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        topic = topic.to(device)

        t = self(dialogue, scene_before, scene_after)
        
        loss_t = self.calculate_loss(t, topic)
        acc_t = self.calculate_acc(t, topic)

        self.log_metric("val", "acc", acc_t)
        self.log_metric("val", "loss", loss_t)
        return loss_t
    
    def test_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, _, _, topic = split_batch(batch)
        
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        topic = topic.to(device)

        t = self(dialogue, scene_before, scene_after)
        
        loss_t = self.calculate_loss(t, topic)
        acc_t = self.calculate_acc(t, topic)

        self.log_metric("test", "acc", acc_t)
        self.log_metric("test", "loss", loss_t)
        return loss_t  

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr) 
    
    def calculate_loss(self, t, topic):
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss_t = self.loss_bce_t(t, topic.float()) + l1_lambda * l1_norm
        return loss_t
        
    def calculate_acc(self, t, topic):        
        acc_t = compute_accuracy_multi_label(t, topic.float())
        return acc_t
    
    def log_metric(self, step, metric, t):
        self.log(f'{step}_{metric}_t', t)
        return
    
    
class MModel(pl.LightningModule):
    def __init__(self, vocab, config):
        super(MModel, self).__init__()
        
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_scenes = config["use_scenes"]
        self.use_instructions = config["use_instructions"]

        self.loss_bce_m = BCEWithLogitsLoss(pos_weight=torch.tensor(MOOD_WEIGHTS), reduction=REDUCTION)
        
        mood_decoder_size = (
            config["decoder_size"] if self.use_instructions and not self.use_scenes
            else config["decoder_size"] * 2 if not self.use_instructions and self.use_scenes
            else config["decoder_size"] * 3 if self.use_instructions and self.use_scenes
            else print("No input available"))
        

        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.dialogue_encoder = EmbeddingCompresser(config["decoder_size"])
        self.mood_decoder = MultiProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
        
        self.scene_compresser = EmbeddingCompresserCNNScene(config["decoder_size"], config["hidden_dim"], config["decoder_size"])
        self.dialogue_compresser = EmbeddingCompresserCNNDialogue(config["hidden_dim"], config["decoder_size"])

        self.save_hyperparameters()
        self.to(device)
        

    def forward(self, dialogue, scene_before, scene_after):

        features1D = []
        if self.use_instructions:
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            features1D.append(self.scene_compresser(scene_before_features)) 
            features1D.append(self.scene_compresser(scene_after_features))

        features = torch.cat(features1D, dim=1).to(device)
        m = self.mood_decoder(features)
        return m
    
    def training_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, mood, _, _ = split_batch(batch)
        
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        mood = mood.to(device)

        m = self(dialogue, scene_before, scene_after)
        
        loss_m = self.calculate_loss(m, mood)

        self.log_metric("train", "loss", loss_m)
        return loss_m

    def validation_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, mood, _, _ = split_batch(batch)

        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        mood = mood.to(device)

        m = self(dialogue, scene_before, scene_after)
        
        loss_m = self.calculate_loss(m, mood)
        acc_m = self.calculate_acc( m, mood)

        self.log_metric("val", "acc", acc_m)
        self.log_metric("val", "loss", loss_m)
        
        return loss_m 
    
    def test_step(self, batch, batch_idx):
        _, _, dialogue, scene_before, scene_after, _, mood, _, _ = split_batch(batch)
        
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        mood = mood.to(device)

        m = self(dialogue, scene_before, scene_after)
        
        loss_m = self.calculate_loss(m, mood)
        acc_m = self.calculate_acc(m, mood)

        self.log_metric("test", "acc", acc_m)
        self.log_metric("test", "loss", loss_m)
        return loss_m 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr) 
    
    def calculate_loss(self, m, mood):
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss_m = self.loss_bce_m(m, mood.float()) + l1_lambda * l1_norm
        return loss_m
        
    def calculate_acc(self, m, mood):        
        acc_m = compute_accuracy_multi_label(m, mood.float()) 
        return acc_m
    
    def log_metric(self, step, metric, m):
        self.log(f'{step}_{metric}_m', m)
        return