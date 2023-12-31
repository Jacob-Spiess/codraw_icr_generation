#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the experiment with a PytorchLightining model.
"""

from argparse import Namespace
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn, Tensor, optim
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

from icr.aux import mask_pads, split_batch, compute_accuracy_text, compute_accuracy_single_label, compute_accuracy_multi_label
from icr.constants import REDUCTION, SPLITS, BOS, EOS, PAD
from icr.evaluator import Metrics, Outputs
from icr import constants
from icr.models import (TextEncoder, TextDecoder, EmbeddingCompresser, SingleProbDecoder, MultiProbDecoder, Attention, 
                        AttentionTextDecoder, TransformerDecoder, EmbeddingExtender)
from icr.components import (
    ActionsMaker, CrossEncoder, iCRClipDecoder, iCRTurnDecoder,
    SceneEncoder, SelfCrossEncoder, StateEmbedding, LearnableLossWeights)


class ICRModel(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel, self).__init__()
        
        self.vocab_size = len(vocab)
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
        
        num_clip_decoder_size = config["decoder_size"]
        clipart_decoder_size = num_clip_decoder_size + config["num_clip_classes"] if self.use_num_clip_decoder else num_clip_decoder_size
        topic_decoder_size = clipart_decoder_size + config["clipart_classes"] if self.use_clipart_decoder else clipart_decoder_size
        mood_decoder_size = topic_decoder_size + config["topic_classes"] if self.use_topic_decoder else topic_decoder_size
        context_size = mood_decoder_size + config["mood_classes"] if self.use_mood_decoder else mood_decoder_size

        self.text_encoder = TextEncoder(self.vocab_size, config["hidden_dim"], config["decoder_size"], drop)
        self.dialogue_encoder = EmbeddingCompresser(config["decoder_size"])
        self.num_clip_decoder = (SingleProbDecoder(num_clip_decoder_size, config["hidden_dim"], config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.n_mapping = EmbeddingExtender(config["num_clip_classes"], config["decoder_size"])
        self.clipart_decoder = (MultiProbDecoder(clipart_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.c_mapping = EmbeddingExtender(config["clipart_classes"], config["decoder_size"])
        self.topic_decoder = (MultiProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.t_mapping = EmbeddingExtender(config["topic_classes"], config["decoder_size"])
        self.mood_decoder = (MultiProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
                                if self.use_mood_decoder
                                else None)
        self.m_mapping = EmbeddingExtender(config["mood_classes"], config["decoder_size"])

        self.decoder = TransformerDecoder(config["decoder_size"], config["hidden_dim_trf"], self.vocab_size, config["nheads"], 
                                          config["nlayers"], drop)
        
        self.save_hyperparameters()


    def forward(self, input_sequence, dialogue, clip, mood, num_clip, topic):
        
        dialogue_features = self.dialogue_encoder(dialogue).unsqueeze(1)
        features = dialogue_features.squeeze(1)
        embeds = self.text_encoder(input_sequence)

        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
            n_features = self.n_mapping(n).unsqueeze(1)
        else:
            n = None
            
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
            c_features = self.c_mapping(c).unsqueeze(1)
        else:
            c = None
        
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
            t_features = self.t_mapping(t).unsqueeze(1)
        else:
            t = None
        
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
            m_features = self.m_mapping(m).unsqueeze(1)
        else:
            m = None
            
        features = torch.cat((dialogue_features, n_features, c_features, t_features, m_features), dim=1)
        outputs = self.decoder(embeds, features)
        
        return outputs, n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)
        
        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_metric("train", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        acc, acc_n, acc_c, acc_t, acc_m = self.calculate_acc(outputs, targets, n, num_clip, c, clip, t, topic, 
                                                             m, mood,self.vocab.stoi[PAD])

        self.log_metric("val", "acc", acc, acc_n, acc_c, acc_t, acc_m)
        self.log_metric("val", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        acc, acc_n, acc_c, acc_t, acc_m = self.calculate_acc(outputs, targets, n, num_clip, c, clip, t, topic, 
                                                             m, mood,self.vocab.stoi[PAD])

        self.log_metric("test", "acc", acc, acc_n, acc_c, acc_t, acc_m)
        self.log_metric("test", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr) #, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        return [optimizer], [lr_scheduler] #optimizer
    
    def calculate_loss(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood):
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        
        loss = self.loss_cross(outputs, targets)
        loss_n = self.loss_cross(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        loss_c = self.loss_bce(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        loss_t = self.loss_bce(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        loss_m = self.loss_bce(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
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

    def reply(self, dialogue, max_len = 39):
        word = torch.tensor(self.vocab.stoi[BOS]).view(1,-1)
        embeds = self.text_encoder(word)
        
        dialogue_features = self.dialogue_encoder(dialogue).unsqueeze(1)
        features = dialogue_features.squeeze(1)
        
        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
            n_features = self.n_mapping(n).unsqueeze(1)
        else:
            n = None
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
            c_features = self.c_mapping(c).unsqueeze(1)
        else:
            c = None
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
            t_features = self.t_mapping(t).unsqueeze(1)
        else:
            t = None
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
            m_features = self.m_mapping(m).unsqueeze(1)
        else:
            m = None
        features = torch.cat((dialogue_features, n_features, c_features, t_features, m_features), dim=1)
        reply = []
        outputs = []
        outputst = torch.zeros(1,max_len, self.vocab_size)

        for i in range(max_len):
            h = self.decoder.transformer(embeds, features)
            output = self.decoder.fc(self.decoder.dropout(h)).view(embeds.size(0),-1)
            outputs.append(output.view(-1, output.size(-1)))
            prediction_index = output.argmax(dim=1)
            reply.append(self.vocab.itos[prediction_index.item()])

            if self.vocab.itos[prediction_index.item()] == EOS:
                l = max_len - i -1
                outputs = torch.stack(outputs, dim = 1)
                outputst = torch.cat((torch.tensor(outputs), torch.zeros(1, l, self.vocab_size)), dim=1)
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))

        return reply, outputst, n, c, t, m

    
class ICRModel1(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel1, self).__init__()
        
        self.vocab_size = len(vocab)
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

        self.text_encoder = TextEncoder(self.vocab_size, config["hidden_dim"], config["word_embedding_size"], drop)
        self.dialogue_encoder = EmbeddingCompresser(config["dialogue_embedding_size"])
        self.num_clip_decoder = (SingleProbDecoder(num_clip_decoder_size, config["hidden_dim"],config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.clipart_decoder = (MultiProbDecoder(clip_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.topic_decoder = (MultiProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.mood_decoder = (MultiProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
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
        
        acc = compute_accuracy_text(outputs, targets, self.vocab.stoi[PAD])
        self.log('val_acc', acc, prog_bar=True)
        self.log_loss("val", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        acc = compute_accuracy_text(outputs, targets, self.vocab.stoi[PAD])
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

        return reply, n, c, t, m
    


class ICRModel2(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel2, self).__init__()
        
        self.vocab_size = len(vocab)
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

        self.text_encoder = TextEncoder(self.vocab_size, config["hidden_dim"], config["word_embedding_size"], drop)
        self.dialogue_encoder = EmbeddingCompresser(config["dialogue_embedding_size"])
        self.num_clip_decoder = (SingleProbDecoder(num_clip_decoder_size, config["hidden_dim"],config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.clipart_decoder = (MultiProbDecoder(clip_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.topic_decoder = (MultiProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.mood_decoder = (MultiProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
                                if self.use_mood_decoder
                                else None)
        
        self.decoder = AttentionTextDecoder(config["word_embedding_size"], context_size, config["decoder_size"], config["attention_size"], 
                                            self.vocab_size, vocab, drop)
        #self.decoder = TextDecoder(config["word_embedding_size"], context_size, vocab_size, config["nlayers"], drop)
        
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
        
        return outputs[0], n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)
        
        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        outputs = outputs.to('cuda:0')
        targets = targets.to('cuda:0')
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_loss("train", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)

        outputs = outputs.to('cuda:0')
        targets = targets.to('cuda:0')
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        
        acc = compute_accuracy_text(outputs, targets, self.vocab.stoi[PAD])
        self.log('val_acc', acc, prog_bar=True)
        self.log_loss("val", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        outputs = outputs.to('cuda:0')
        targets = targets.to('cuda:0')
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        acc = compute_accuracy_text(outputs, targets, self.vocab.stoi[PAD])
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
        
        h, c = self.decoder.init_hidden_state(features)
        
        reply = []

        for _ in range(max_len):
            attention_weighted_encoding, alpha = self.decoder.attention(features, h)
            gate = self.decoder.sigmoid(self.decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            lstm_input = torch.cat((embeds[:, 0], attention_weighted_encoding), dim=1)
            h, c = self.decoder.lstm_cell(lstm_input, (h, c))
            
            output = self.decoder.fcn(self.decoder.drop(h)).view(embeds.size(0),-1)

            prediction_index = output.argmax(dim=1)
            reply.append(self.vocab.itos[prediction_index.item()])

            if self.vocab.itos[prediction_index.item()] == EOS:
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))

        return reply, n, c, t, m