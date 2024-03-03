"""
Evaluator based on https://github.com/StanfordMIMI/RaLEs/blob/5cc18cee31f2be970acc23782aa203babd9b82d6/evaluation/report_generation_evaluator.py
Distinct-n Scorer based on https://github.com/lancopku/text-autoaugment/blob/a74d30b07b1004367a2d86dd38396d55c80d6d8b/taa/utils/distinct_n.py#L88
"""

from itertools import chain
import nltk
nltk.download('wordnet')
import torch
import torch.nn as nn
from torchmetrics.text.bert import BERTScore
from torchmetrics.text import CHRFScore
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from aac_metrics.classes.cider_d import CIDErD
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu

from icr.aux import list_avg

class BertScore(nn.Module):
    def __init__(self, digits = 4):
        super(BertScore, self).__init__()
        self.digits = digits
        with torch.no_grad():
            self.bert_scorer = BERTScore(model_name_or_path='distilbert-base-uncased', num_layers=5, batch_size=64,
                                          num_threads=1, all_layers=False, idf=False, 
                                          device='cuda' if torch.cuda.is_available() else 'cpu', lang='en',
                                          rescale_with_baseline=True, baseline_path=None)

    def forward(self, refs, hyps):
        scores= self.bert_scorer(preds=hyps, target=refs)
        return round(torch.mean(scores["f1"]).item(), self.digits), scores["f1"].tolist()

class ChrfScore(nn.Module):
    def __init__(self, digits = 4):
        super(ChrfScore, self).__init__()
        self.digits = digits
        with torch.no_grad():
            self.chrf_scorer = CHRFScore(n_word_order=0, return_sentence_level_score=True)

    def forward(self, refs, hyps):
        scores= self.chrf_scorer(hyps, refs)
        return round(scores[0].item(), self.digits), scores[1].tolist()

class BleuScore(nn.Module):
    def __init__(self, weights, digits = 4):
        super(BleuScore, self).__init__()
        self.digits = digits
        self.weights = weights
            
    def forward(self, refs, hyps):
        bleu_scores = []
        for hypothesis, reference in zip(hyps, refs):
            hypothesis_tokens = nltk.word_tokenize(hypothesis)
            reference_tokens = [nltk.word_tokenize(reference)]

            score = sentence_bleu(reference_tokens, hypothesis_tokens, weights=self.weights)
            bleu_scores.append(score)
        average_bleu_score = sum(bleu_scores) / len(bleu_scores)
        return round(average_bleu_score, self.digits), bleu_scores
    
class RogueScore(nn.Module):
    def __init__(self, metric = "rouge1_fmeasure", digits = 4):
        super(RogueScore, self).__init__()
        self.metric = metric
        self.digits = digits
        with torch.no_grad():
            self.rogue_scorer = ROUGEScore()
            
    def forward(self, refs, hyps):
        scores = []
        if len(refs) == len(hyps):
            for i, _ in enumerate(hyps):
                scores.append(self.rogue_scorer(hyps[i], refs[i])[self.metric].item())
        else:
            scores.append(0)
        return round(list_avg(scores), self.digits), scores
    
class CiderScore(nn.Module):
    def __init__(self, digits = 4):
        super(CiderScore, self).__init__()
        self.digits = digits
        with torch.no_grad():
            self.cider_scorer = CIDErD()
            
    def forward(self, refs, hyps):
        refs = [[item] for item in refs]
        scores= self.cider_scorer(hyps, refs)
        return round(scores[0]['cider_d'].item(), self.digits), scores[1]['cider_d'].tolist()
    
class MeteorScore(nn.Module):
    def __init__(self, digits = 4):
        super(MeteorScore, self).__init__()
        self.digits = digits
            
    def forward(self, refs, hyps):
        scores = []
        if len(refs) == len(hyps):
            for i, _ in enumerate(hyps):
                scores.append(single_meteor_score(refs[i].split(), hyps[i].split()))
        else:
            scores.append(0)
        return round(list_avg(scores), self.digits), scores

    
class DistinctScore(nn.Module):
    def __init__(self, n = 1, digits = 4):
        super(DistinctScore, self).__init__()
        self.digits = digits
        self.n = n
            
    def forward(self, refs, hyps):
        hyps =[sentences.split() for sentences in hyps]
        score = self.distinct_n_corpus_level(hyps, self.n)
        return round(score, self.digits), 0
    
    def distinct_n_corpus_level(self, sentences, n):
        return sum(self.distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)

    def distinct_n_sentence_level(self, sentence, n):
        if len(sentence) == 0:
            return 0.0  # Prevent a zero division
        distinct_ngrams = set(self.ngrams(sentence, n))
        return len(distinct_ngrams) / len(sentence)

    def ngrams(self, sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
        sequence = self.pad_sequence(sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol)
        history = []
        while n > 1:
            history.append(next(sequence))
            n -= 1
        for item in sequence:
            history.append(item)
            yield tuple(history)
            del history[0]

    def pad_sequence(self, sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
        sequence = iter(sequence)
        if pad_left:
            sequence = chain((left_pad_symbol,) * (n - 1), sequence)
        if pad_right:
            sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
        return sequence

    
SCORER_NAME_TO_CLASS = {
    "BERT Score": BertScore(),
    "ChrF Score": ChrfScore(),
    "BLEU Score1": BleuScore((1, 0, 0, 0)),
    "BLEU Score2": BleuScore((0.5, 0.5, 0, 0)),
    "BLEU Score3": BleuScore((0.33, 0.33, 0.33, 0)),
    "BLEU Score4": BleuScore((0.25, 0.25, 0.25, 0.25)),
    "ROUGE Score F1": RogueScore("rouge1_fmeasure"),
    "CIDEr Score": CiderScore(),
    "METEOR Score": MeteorScore(),
    "Distinct-1 Score": DistinctScore(n = 1),
    "Distinct-2 Score": DistinctScore(n = 2)
}


class GenerationEvaluator:
    def __init__(self, scorers=['BERT Score', "ChrF Score", "BLEU Score1", "BLEU Score2", "BLEU Score3", "BLEU Score4", "ROUGE Score F1",
                               "CIDEr Score", "METEOR Score", "Distinct-1 Score", "Distinct-2 Score"]):
        self.scorers = {}

        for scorer_name in scorers:
            #if scorer_name.lower() in SCORER_NAME_TO_CLASS:
            if scorer_name in SCORER_NAME_TO_CLASS: 
                self.scorers[scorer_name] = SCORER_NAME_TO_CLASS[scorer_name]  
            else:
                raise NotImplementedError(f'scorer of type {scorer_name} not implemented')

    def evaluate(self, hypotheses, references):
        assert len(hypotheses) == len(references), f'Length of hypotheses (i.e. generations) {len(hypotheses)} and references (i.e. ground truths) {len(references)} must match. '
        scores = {k:None for k in self.scorers.keys()}
        for scorer_name, scorer in self.scorers.items():
            scores[scorer_name] = scorer(refs=references, hyps=hypotheses)[0]

        return scores