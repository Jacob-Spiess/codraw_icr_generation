from typing import Dict, Tuple, List
import spacy
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from icr import aux
from icr.constants import UNK, PAD, BOS, EOS#, spacy_eng



class Vocabulary:
    #spacy_eng = en_core_web_sm.load()
    
    def __init__(self, codraw_path, freq_threshold=2):
        self.itos = {0: PAD, 1: BOS, 2: EOS, 3: UNK}
        self.stoi = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
        self.freq_threshold = freq_threshold
        
        self.codraw = self.load_codraw(codraw_path)
        
        self.dialogues, self.max_token = self.extract_dialogue_info()
                
        self.build_vocabulary(self.dialogues)
        print("Vocabulary size is " + str(len(self.itos)))        

    def __len__(self): 
        return len(self.itos)
    
    #@staticmethod
    #def tokenize(text):
    #    return [token.text.lower() for token in spacy_eng.tokenizer(text)] #Vocabulary.
    
    def build_vocabulary(self, dialogues):
        frequencies = Counter()
        idx = 4
        for token in dialogues.split():
            frequencies[token] += 1
            if frequencies[token] == self.freq_threshold:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1
        print("Vocabulary was successfully built!")
    
    def numericalize(self, utterance):
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in utterance]    
    
    def extract_dialogue_info(self):
        all_dialogues = ""
        utterances = []
        for name, dialogue in tqdm(self.codraw["data"].items()):
            for turn in dialogue['dialog']:
                all_dialogues += " " + turn["msg_t"]
                all_dialogues += " " + turn["msg_d"]
                utterances += [turn["msg_t"]] + [turn["msg_d"]]
                
        max_len = max(len(u.split()) for u in utterances)
        return all_dialogues, max_len
    
    def load_codraw(self, codraw_path):
        
        with open(Path(codraw_path), 'r') as file:
            codraw = json.load(file)
        return codraw