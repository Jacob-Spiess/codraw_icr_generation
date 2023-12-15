#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the experiment with a PytorchLightining model.
"""

from argparse import Namespace
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

from icr.aux import mask_pads, split_batch, compute_accuracy #filter_checkpoint
from icr.constants import REDUCTION, SPLITS, BOS, EOS, PAD
from icr.evaluator import Metrics, Outputs
from icr import constants
from icr.models import TextEncoder, TextDecoder, EmbeddingCompresser, ProbDecoder
from icr.components import (
    ActionsMaker, CrossEncoder, iCRClipDecoder, iCRTurnDecoder,
    SceneEncoder, SelfCrossEncoder, StateEmbedding, LearnableLossWeights)

   
    
class ICRModel1(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel1, self).__init__()
        
        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        max_length = vocab.max_token
        self.vocab = vocab
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_num_clip_decoder = config["use_num_clip_decoder"]
        self.use_clipart_decoder = config["use_clipart_decoder"]
        self.use_topic_decoder = config["use_topic_decoder"]
        self.use_mood_decoder = config["use_mood_decoder"]

        self.loss_cross = CrossEntropyLoss(reduction=REDUCTION)
        self.loss_bce = BCEWithLogitsLoss(reduction=REDUCTION)
        
        num_clip_decoder_size = config["dialogue_embedding_size"]
        clip_decoder_size = num_clip_decoder_size + config["num_clip_classes"] if self.use_num_clip_decoder else num_clip_decoder_size
        topic_decoder_size = clip_decoder_size + config["clipart_classes"] if self.use_clipart_decoder else clip_decoder_size
        mood_decoder_size = topic_decoder_size + config["topic_classes"] if self.use_topic_decoder else topic_decoder_size
        context_size = mood_decoder_size + config["mood_classes"] if self.use_mood_decoder else mood_decoder_size

        self.text_encoder = TextEncoder(vocab_size, config["hidden_dim"], config["word_embedding_size"], drop)
        self.dialogue_encoder = EmbeddingCompresser(config["dialogue_embedding_size"])
        self.num_clip_decoder = (ProbDecoder(num_clip_decoder_size, config["hidden_dim"],config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.clipart_decoder = (ProbDecoder(clip_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.topic_decoder = (ProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.mood_decoder = (ProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
                                if self.use_mood_decoder
                                else None)

        self.decoder = TextDecoder(config["word_embedding_size"], context_size, vocab_size, config["nlayers"], drop)
        
        self.save_hyperparameters()


    def forward(self, input_sequence, dialogue, clip, mood, num_clip, topic):
        
        features = self.dialogue_encoder(dialogue)    
        embeds = self.text_encoder(input_sequence)

        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
        else:
            n = None
            
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
        else:
            c = None
        
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
        else:
            t = None
        
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
        else:
            m = None
        
        outputs = self.decoder(embeds, features)
        
        return outputs, n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)
        
        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_loss("train", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)

        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        
        acc = compute_accuracy(outputs, targets, self.vocab.stoi[PAD])
        self.log('val_acc', acc, prog_bar=True)
        self.log_loss("val", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        acc = compute_accuracy(outputs, targets, self.vocab.stoi[PAD])
        self.log('test_acc', acc, prog_bar=True)
        self.log_loss("test", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def calculate_loss(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood):
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        
        loss = self.loss_cross(outputs, targets)
        loss_n = self.loss_cross(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        loss_c = self.loss_bce(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        loss_t = self.loss_bce(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        loss_m = self.loss_bce(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return loss, loss_n, loss_c, loss_t, loss_m
        
    def log_loss(self, step, loss, loss_n, loss_c, loss_t, loss_m):
        self.log(f'{step}_loss', loss)
        self.log(f'{step}_loss_n', loss_n)
        self.log(f'{step}_loss_c', loss_c)
        self.log(f'{step}_loss_t', loss_t)
        self.log(f'{step}_loss_m', loss_m)
        return

    def reply(self, dialogue, max_len = 40):
        word = torch.tensor(self.vocab.stoi[BOS]).view(1,-1)
        embeds = self.text_encoder(word)
        
        features = self.dialogue_encoder(dialogue)
        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
        
        h0 = features.repeat(self.decoder.num_layers, 1, 1) 
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        reply = []

        for _ in range(max_len):
            x, hidden = self.decoder.lstm(embeds, hidden)
            output = self.decoder.fc(x).view(embeds.size(0),-1)

            prediction_index = output.argmax(dim=1)
            reply.append(self.vocab.itos[prediction_index.item()])

            if self.vocab.itos[prediction_index.item()] == EOS:
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))

        return reply
    

class ICRModel2(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel1, self).__init__()
        
        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        max_length = vocab.max_token
        self.vocab = vocab
        drop = config["dropout"]

        self.loss_cross = CrossEntropyLoss(reduction=REDUCTION)
        self.loss_bce = BCEWithLogitsLoss(reduction=REDUCTION)
        
        num_clip_decoder_size = config["dialogue_embedding_size"]
        clip_decoder_size = num_clip_decoder_size + config["num_clip_classes"]
        topic_decoder_size = clip_decoder_size + config["clipart_classes"]
        mood_decoder_size = topic_decoder_size + config["topic_classes"]
        context_size = mood_decoder_size + config["mood_classes"]

        self.text_encoder = TextEncoder(vocab_size, config["hidden_dim"], config["word_embedding_size"], drop)
        self.dialogue_encoder = EmbeddingCompresser(config["dialogue_embedding_size"])
        self.num_clip_decoder = ProbDecoder(num_clip_decoder_size, config["hidden_dim"], config["num_clip_classes"], drop)
        self.clipart_decoder = ProbDecoder(clip_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
        self.topic_decoder = ProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
        self.mood_decoder = ProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)

        self.decoder = TextDecoder(config["word_embedding_size"], context_size, vocab_size, config["nlayers"], drop)
        
        self.save_hyperparameters()


    def forward(self, input_sequence, dialogue, clip, mood, num_clip, topic):
        
        features = self.dialogue_encoder(dialogue)    
        embeds = self.text_encoder(input_sequence)

        n = self.num_clip_decoder(features)
        features = torch.cat((features, n), dim=1)
        
        c = self.clipart_decoder(features)
        features = torch.cat((features, c), dim=1)
        
        t = self.topic_decoder(features)
        features = torch.cat((features, t), dim=1)
        
        m = self.mood_decoder(features)
        features = torch.cat((features, m), dim=1)
        
        outputs = self.decoder(embeds, features)
        
        return outputs, n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)
        
        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_loss("train", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        self.eval()  # Ensure the model is in evaluation mode
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)

        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        acc = (outputs.view(-1, outputs.size(-1)).argmax(dim=1) == targets.reshape(-1)).sum().item() / len(targets.reshape(-1))
        self.log('val_acc', acc, prog_bar=True)
        self.log_loss("val", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        acc = (outputs.view(-1, outputs.size(-1)).argmax(dim=1) == targets.reshape(-1)).sum().item() / len(targets.reshape(-1))
        self.log('test_acc', acc, prog_bar=True)
        self.log_loss("test", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.0001)
    
    def calculate_loss(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood):
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        
        loss = self.loss_cross(outputs, targets)
        loss_n = self.loss_cross(n, num_clip.argmax(dim=1))
        loss_c = self.loss_bce(c, clip.float())
        loss_t = self.loss_bce(t, topic.float())
        loss_m = self.loss_bce(m, mood.float())
        
        return loss, loss_n, loss_c, loss_t, loss_m
        
    def log_loss(self, step, loss, loss_n, loss_c, loss_t, loss_m):
        self.log(f'{step}_loss', loss)
        self.log(f'{step}_loss_n', loss_n)
        self.log(f'{step}_loss_c', loss_c)
        self.log(f'{step}_loss_t', loss_t)
        self.log(f'{step}_loss_m', loss_m)
        return

    def reply(self, dialogue, max_len = 40):
        word = torch.tensor(self.vocab.stoi[BOS]).view(1,-1)
        embeds = self.text_encoder(word)
        
        features = self.dialogue_encoder(dialogue)
        n = self.num_clip_decoder(features)
        features = torch.cat((features, n), dim=1)
        c = self.clipart_decoder(features)
        features = torch.cat((features, c), dim=1)
        t = self.topic_decoder(features)
        features = torch.cat((features, t), dim=1)
        m = self.mood_decoder(features)
        features = torch.cat((features, m), dim=1)
        
        h0 = features.repeat(self.decoder.num_layers, 1, 1) 
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        reply = []

        for _ in range(max_len):
            x, hidden = self.decoder.lstm(embeds, hidden)
            output = self.decoder.fc(x).view(embeds.size(0),-1)

            prediction_index = output.argmax(dim=1)
            reply.append(self.vocab.itos[prediction_index.item()])

            if self.vocab.itos[prediction_index.item()] == EOS:
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))

        return reply
