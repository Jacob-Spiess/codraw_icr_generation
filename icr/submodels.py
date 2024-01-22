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
from transformers import BartTokenizer


from icr.aux import (mask_pads, split_batch, compute_accuracy_text, compute_accuracy_single_label, 
                     compute_accuracy_multi_label, load_partial_state_dict)
from icr.constants import REDUCTION, SPLITS, BOS, EOS, PAD, NUM_WEIGHTS, CLIP_WEIGHTS, TOPIC_WEIGHTS, MOOD_WEIGHTS
from icr.evaluator import Metrics, Outputs
from icr import constants
from icr.models import (TextEncoder, TextDecoder, EmbeddingCompresser, EmbeddingCompresserLarge, SingleProbDecoder, MultiProbDecoder, 
                        Attention, AttentionTextDecoder, TransformerDecoder, EmbeddingExtender, SceneEncoder, 
                        EmbeddingCompresserCNNScene, EmbeddingCompresserCNNDialogue, BartTransformerDecoder)


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
    
    
class ICRModelpretrained(pl.LightningModule):
    def __init__(self, vocab, config, 
                 num_clip_decoder_state=None, clip_decoder_state=None, topic_decoder_state=None, mood_decoder_state=None):
        super(ICRModelpretrained, self).__init__()
        
        self.vocab_size = len(vocab)
        max_length = vocab.max_token
        self.vocab = vocab
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_scenes = config["use_scenes"]
        self.use_instructions = config["use_instructions"]
        self.use_num_clip_decoder = config["use_num_clip_decoder"]
        self.use_clipart_decoder = config["use_clipart_decoder"]
        self.use_topic_decoder = config["use_topic_decoder"]
        self.use_mood_decoder = config["use_mood_decoder"]

        self.loss_cross = CrossEntropyLoss(reduction=REDUCTION)
        self.loss_cross_n = CrossEntropyLoss(weight=torch.tensor(NUM_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_c = BCEWithLogitsLoss(pos_weight=torch.tensor(CLIP_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_t = BCEWithLogitsLoss(pos_weight=torch.tensor(TOPIC_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_m = BCEWithLogitsLoss(pos_weight=torch.tensor(MOOD_WEIGHTS), reduction=REDUCTION)
        
        classifier_decoder_size = (
            config["decoder_size"] if self.use_instructions and not self.use_scenes
            else config["decoder_size"] * 2 if not self.use_instructions and self.use_scenes
            else config["decoder_size"] * 3 if self.use_instructions and self.use_scenes
            else print("No input available"))

        self.text_encoder = TextEncoder(self.vocab_size, config["hidden_dim"], config["decoder_size"], drop)
        self.pe = PositionalEncoding1D(config["decoder_size"])
        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.dialogue_encoder = EmbeddingCompresser(config["decoder_size"])
        self.num_clip_decoder = (SingleProbDecoder(classifier_decoder_size, config["hidden_dim"], config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.n_mapping = EmbeddingExtender(config["num_clip_classes"], config["decoder_size"])
        self.clipart_decoder = (MultiProbDecoder(classifier_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.c_mapping = EmbeddingExtender(config["clipart_classes"], config["decoder_size"])
        self.topic_decoder = (MultiProbDecoder(classifier_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.t_mapping = EmbeddingExtender(config["topic_classes"], config["decoder_size"])
        self.mood_decoder = (MultiProbDecoder(classifier_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
                                if self.use_mood_decoder
                                else None)
        self.m_mapping = EmbeddingExtender(config["mood_classes"], config["decoder_size"])

        self.decoder = TransformerDecoder(config["decoder_size"], config["hidden_dim_trf"], self.vocab_size, config["nheads"], 
                                          config["nlayers"], drop)
        self.scene_compresser = EmbeddingCompresserCNNScene(config["decoder_size"], config["hidden_dim"], config["decoder_size"])
        self.dialogue_compresser = EmbeddingCompresserCNNDialogue(config["hidden_dim"], config["decoder_size"])

        #Load existing parameters for number, clipart, topic and mood classifer
        if num_clip_decoder_state:
            load_partial_state_dict(num_clip_decoder, torch.load(num_clip_decoder_state), 'num_clip_decoder.')
            for param in num_clip_decoder.parameters():
                param.requires_grad = False
        if clip_decoder_state:
            load_partial_state_dict(clipart_decoder, torch.load(clip_decoder_state), 'clipart_decoder.')
            for param in self.clipart_decoder.parameters():
                param.requires_grad = False
        if topic_decoder_state:
            load_partial_state_dict(topic_decoder, torch.load(topic_decoder_state), 'topic_decoder.')
            for param in self.topic_decoder.parameters():
                param.requires_grad = False
        if mood_decoder_state:
            load_partial_state_dict(mood_decoder, torch.load(mood_decoder_state), 'mood_decoder.')
            for param in self.mood_decoder.parameters():
                param.requires_grad = False
                
        self.save_hyperparameters()
        self.to(device)
        

    def forward(self, input_sequence, dialogue, scene_before, scene_after):

        features1D = []
        features2D = []
        if self.use_instructions:
            dialogue_features = self.dialogue_encoder(dialogue)
            positions = mask_pads(dialogue_features, self.pe(dialogue_features))
            dialogue_features = torch.cat([dialogue_features + positions],dim=1)
            features2D.append(dialogue_features)
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            scenes = torch.cat((scene_before_features, scene_after_features), dim=1)
            positions = self.pe(scenes)
            scene_features = torch.cat([scenes + positions], dim=1)
            features2D.append(scene_features)
            features1D.append(self.scene_compresser(scene_before_features)) 
            features1D.append(self.scene_compresser(scene_after_features)) 

        features = torch.cat(features1D, dim=1).to(device)
        padding_mask = input_sequence == self.vocab.stoi[PAD]
        embeds = self.text_encoder(input_sequence).to(device)

        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            n_features = self.n_mapping(n).unsqueeze(1)
            features2D.append(n_features)
        else:
            n = None
            
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            c_features = self.c_mapping(c).unsqueeze(1)
            features2D.append(c_features)
        else:
            c = None
        
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            t_features = self.t_mapping(t).unsqueeze(1)
            features2D.append(t_features)
        else:
            t = None
        
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            m_features = self.m_mapping(m).unsqueeze(1)
            features2D.append(m_features)
        else:
            m = None

        features = torch.cat(features2D, dim=1).to(device)
        outputs = self.decoder(embeds, features, padding_mask)
        
        return outputs, n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_metric("train", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        acc, acc_n, acc_c, acc_t, acc_m = self.calculate_acc(outputs, targets, n, num_clip, c, clip, t, topic, 
                                                             m, mood,self.vocab.stoi[PAD])

        self.log_metric("val", "acc", acc, acc_n, acc_c, acc_t, acc_m)
        self.log_metric("val", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        acc, acc_n, acc_c, acc_t, acc_m = self.calculate_acc(outputs, targets, n, num_clip, c, clip, t, topic, 
                                                             m, mood,self.vocab.stoi[PAD])

        self.log_metric("test", "acc", acc, acc_n, acc_c, acc_t, acc_m)
        self.log_metric("test", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr) 
        return optimizer 
    
    def calculate_loss(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood):
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())

        loss = self.loss_cross(outputs, targets) + l1_lambda * l1_norm
        loss_n = self.loss_cross_n(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        loss_c = self.loss_bce_c(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        loss_t = self.loss_bce_t(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        loss_m = self.loss_bce_m(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return loss, loss_n, loss_c, loss_t, loss_m
        
    def calculate_acc(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood, pad):        
        acc = compute_accuracy_text(outputs, targets, pad)
        acc_n = compute_accuracy_single_label(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        acc_c = compute_accuracy_multi_label(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        acc_t = compute_accuracy_multi_label(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        acc_m = compute_accuracy_multi_label(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return acc, acc_n, acc_c, acc_t, acc_m
    
    def log_metric(self, step, metric, total, n, c, t, m):
        self.log(f'{step}_{metric}', total)
        self.log(f'{step}_{metric}_n', n)
        self.log(f'{step}_{metric}_c', c)
        self.log(f'{step}_{metric}_t', t)
        self.log(f'{step}_{metric}_m', m)
        return

    def top_k_top_p_filtering(self, logits, top_k, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
            
        return logits

    def reply(self, dialogue, scene_before, scene_after, use_sampling: bool, top_k: int, top_p: float, temperature: float, max_len = 39):
        self.text_encoder.to(device)
        self.num_clip_decoder.to(device)
        self.clipart_decoder.to(device)
        self.topic_decoder.to(device)
        self.mood_decoder.to(device)
        self.n_mapping.to(device), self.c_mapping.to(device), self.t_mapping.to(device), self.m_mapping.to(device)
        self.decoder.to(device)
        dialogue.to(device)
        scene_before.to(device)
        scene_after.to(device)
        
        word = torch.tensor(self.vocab.stoi[BOS]).view(1,-1).to(device)
        embeds = self.text_encoder(word)
        
        features1D = []
        features2D = []
        if self.use_instructions:
            dialogue_features = self.dialogue_encoder(dialogue)
            positions = mask_pads(dialogue_features, self.pe(dialogue_features))
            dialogue_features = torch.cat([dialogue_features + positions],dim=1)
            features2D.append(dialogue_features)
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            scenes = torch.cat((scene_before_features, scene_after_features), dim=1)
            positions = self.pe(scenes)
            scene_features = torch.cat([scenes + positions], dim=1)
            features2D.append(scene_features)
            features1D.append(self.scene_compresser(scene_before_features))
            features1D.append(self.scene_compresser(scene_after_features))

        features = torch.cat(features1D, dim=1).to(device)
              
        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features).to(device)
            n_features = self.n_mapping(n).unsqueeze(1)
            features2D.append(n_features)
        else:
            n = None
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features).to(device)
            c_features = self.c_mapping(c).unsqueeze(1)
            features2D.append(c_features)
        else:
            c = None
        if self.use_topic_decoder:
            t = self.topic_decoder(features).to(device)
            t_features = self.t_mapping(t).unsqueeze(1)
            features2D.append(t_features)
        else:
            t = None
        if self.use_mood_decoder:
            m = self.mood_decoder(features).to(device)
            m_features = self.m_mapping(m).unsqueeze(1)
            features2D.append(m_features)
        else:
            m = None
            
        features = torch.cat(features2D, dim=1).to(device)
        reply = []
        outputs = []
        outputs_tensor = torch.zeros(1,max_len, self.vocab_size).to(device)

        for i in range(max_len):
            h = self.decoder.transformer(embeds, features)
            output = self.decoder.fc(self.decoder.dropout(h)).view(embeds.size(0),-1).to(device)
            outputs.append(output.view(-1, output.size(-1)))
            
            if use_sampling:
                output = output / temperature
                filtered_logits = self.top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                prediction_index = torch.multinomial(probabilities, 1).squeeze(1)
            else:
                prediction_index = output.argmax(dim=1)
            
            reply.append(self.vocab.itos[prediction_index.item()])

            if self.vocab.itos[prediction_index.item()] == EOS:
                l = max_len - i -1
                outputs = torch.stack(outputs, dim = 1).to(device)
                outputs_tensor = torch.cat((outputs.clone().detach(), torch.zeros(1, l, self.vocab_size).to(device)), dim=1)
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))

        return reply, outputs_tensor, n, c, t, m