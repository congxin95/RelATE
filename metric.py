#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import pdb

class TripletMetric:
    def __init__(self):        
        self.pred = []
        self.true = []
        self.cnt = 0

    def update_state(self, batch_preds, batch_trues, batch_id2label):
        batch_size = len(batch_id2label)
        
        _, seq_len = batch_trues.shape
        batch_preds = batch_preds.view(batch_size, -1, seq_len)
        batch_trues = batch_trues.view(batch_size, -1, seq_len)
        
        batch_preds = batch_preds.cpu().tolist()
        batch_trues = batch_trues.cpu().tolist()
        
        for preds, trues, id2label in zip(batch_preds, batch_trues, batch_id2label):
            preds = self.decode(preds, id2label)
            trues = self.decode(trues, id2label)
            
            pred_triplet = self.extract(preds)
            true_triplet = self.extract(trues)
            
            self.pred.extend(pred_triplet)
            self.true.extend(true_triplet) 
            self.cnt += 1
    
    def result(self):
        return self.score(self.pred, self.true)
    
    def reset(self):
        self.pred = []
        self.true = []
        self.cnt = 0

    def decode(self, ids, id2label):
        labels = []
        for ins in ids:
            ins_labels = list(map(lambda x: id2label[x], ins))
            labels.append(ins_labels)
        return labels
    
    def extract(self, label_sequences):
        results = []
        for i, instance_label in enumerate(label_sequences):
            spans = self.get_span(instance_label)
            result = []
            
            relations = {}
            for span in spans:
                relation, entity_type = span[0].split(":")
                
                if relation not in relations:
                    relations[relation] = {"HEAD": [], "TAIL": []}
                    
                relations[relation][entity_type].append(span)
            
            for relation, entities in relations.items():
                heads = entities["HEAD"]
                tails = entities["TAIL"]
                
                for head in heads:
                    for tail in tails:
                        triplet = (self.cnt, i, head, tail)
                        result.append(triplet)
            results.extend(result)
            
        return results
    
    def score(self, pred_tags, true_tags):    
        true_triplets = set(self.pred)
        pred_triplets = set(self.true)
        
        pred_correct = len(true_triplets & pred_triplets)
        pred_all = len(pred_triplets)
        true_all = len(true_triplets)
    
        p = pred_correct / pred_all if pred_all > 0 else 0
        r = pred_correct / true_all if true_all > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        return p, r, f1

    def get_span(self, seq):
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ['O']]
        
        prev_tag = 'O'
        prev_type = ''
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ['O']):
            tag = chunk[0]
            type_ = chunk.split('-')[-1]
    
            if self.end_of_span(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i-1))
            if self.start_of_span(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_
    
        return chunks
    
    def start_of_span(self, prev_tag, tag, prev_type, type_):
        chunk_start = False
    
        if tag == 'B': chunk_start = True
        if tag == 'S': chunk_start = True
    
        if prev_tag == 'E' and tag == 'E': chunk_start = True
        if prev_tag == 'E' and tag == 'I': chunk_start = True
        if prev_tag == 'S' and tag == 'E': chunk_start = True
        if prev_tag == 'S' and tag == 'I': chunk_start = True
        if prev_tag == 'O' and tag == 'E': chunk_start = True
        if prev_tag == 'O' and tag == 'I': chunk_start = True
    
        if tag != 'O' and tag != '.' and prev_type != type_:
            chunk_start = True
    
        return chunk_start
    
    def end_of_span(self, prev_tag, tag, prev_type, type_):
        chunk_end = False
    
        if prev_tag == 'E': chunk_end = True
        if prev_tag == 'S': chunk_end = True
    
        if prev_tag == 'B' and tag == 'B': chunk_end = True
        if prev_tag == 'B' and tag == 'S': chunk_end = True
        if prev_tag == 'B' and tag == 'O': chunk_end = True
        if prev_tag == 'I' and tag == 'B': chunk_end = True
        if prev_tag == 'I' and tag == 'S': chunk_end = True
        if prev_tag == 'I' and tag == 'O': chunk_end = True
    
        if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
            chunk_end = True
    
        return chunk_end

class EntityMetric:
    def __init__(self):        
        self.pred = []
        self.true = []

    def update_state(self, batch_preds, batch_trues, batch_id2label):
        batch_size = len(batch_id2label)
        
        _, seq_len = batch_trues.shape
        batch_preds = batch_preds.view(batch_size, -1, seq_len)
        batch_trues = batch_trues.view(batch_size, -1, seq_len)
        
        batch_preds = batch_preds.cpu().tolist()
        batch_trues = batch_trues.cpu().tolist()
        
        for preds, trues, id2label in zip(batch_preds, batch_trues, batch_id2label):
            preds = self.decode(preds, id2label)
            trues = self.decode(trues, id2label)
            
            self.pred.extend(preds)
            self.true.extend(trues) 
    
    def result(self):
        return self.score(self.pred, self.true)
    
    def reset(self):
        self.pred = []
        self.true = []

    def decode(self, ids, id2label):
        labels = []
        for ins in ids:
            ins_labels = list(map(lambda x: id2label[x], ins))
            labels.append(ins_labels)
        return labels
    
    def score(self, pred_tags, true_tags):
        true_spans = set(self.get_span(true_tags))
        pred_spans = set(self.get_span(pred_tags))
    
        pred_correct = len(true_spans & pred_spans)
        pred_all = len(pred_spans)
        true_all = len(true_spans)
    
        p = pred_correct / pred_all if pred_all > 0 else 0
        r = pred_correct / true_all if true_all > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        return p, r, f1

    def get_span(self, seq):
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ['O']]
        
        prev_tag = 'O'
        prev_type = ''
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ['O']):
            tag = chunk[0]
            type_ = chunk.split(':')[-1]
    
            if self.end_of_span(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i-1))
            if self.start_of_span(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_
    
        return chunks
    
    def start_of_span(self, prev_tag, tag, prev_type, type_):
        chunk_start = False
    
        if tag == 'B': chunk_start = True
        if tag == 'S': chunk_start = True
    
        if prev_tag == 'E' and tag == 'E': chunk_start = True
        if prev_tag == 'E' and tag == 'I': chunk_start = True
        if prev_tag == 'S' and tag == 'E': chunk_start = True
        if prev_tag == 'S' and tag == 'I': chunk_start = True
        if prev_tag == 'O' and tag == 'E': chunk_start = True
        if prev_tag == 'O' and tag == 'I': chunk_start = True
    
        if tag != 'O' and tag != '.' and prev_type != type_:
            chunk_start = True
    
        return chunk_start
    
    def end_of_span(self, prev_tag, tag, prev_type, type_):
        chunk_end = False
    
        if prev_tag == 'E': chunk_end = True
        if prev_tag == 'S': chunk_end = True
    
        if prev_tag == 'B' and tag == 'B': chunk_end = True
        if prev_tag == 'B' and tag == 'S': chunk_end = True
        if prev_tag == 'B' and tag == 'O': chunk_end = True
        if prev_tag == 'I' and tag == 'B': chunk_end = True
        if prev_tag == 'I' and tag == 'S': chunk_end = True
        if prev_tag == 'I' and tag == 'O': chunk_end = True
    
        if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
            chunk_end = True
    
        return chunk_end

class HeadMetric(EntityMetric):
    
    def decode(self, ids, id2label):
        t_id2label = {i: (j if "TAIL" not in j else "O") for i, j in id2label.items()}
        
        labels = []
        for ins in ids:
            ins_labels = list(map(lambda x: t_id2label[x], ins))
            labels.append(ins_labels)
        return labels

class TailMetric(EntityMetric):
    
    def decode(self, ids, id2label):
        t_id2label = {i: (j if "HEAD" not in j else "O") for i, j in id2label.items()}
        
        labels = []
        for ins in ids:
            ins_labels = list(map(lambda x: t_id2label[x], ins))
            labels.append(ins_labels)
        return labels

class RelationMetricV1:
    def __init__(self):
        self.pred = []
        self.true = []

    def update_state(self, batch_preds, batch_trues):        
        batch_preds = batch_preds.cpu().numpy()
        batch_trues = batch_trues.cpu().numpy()
            
        self.pred.append(batch_preds)
        self.true.append(batch_trues)
    
    def result(self):
        return self.score()
    
    def reset(self):
        self.pred = []
        self.true = []
    
    def score(self):
        pred = np.concatenate(self.pred)
        true = np.concatenate(self.true)
        
        p = precision_score(true, pred, average="macro")
        r = recall_score(true, pred, average="macro")
        f1 = f1_score(true, pred, average="macro")

        return p, r, f1

class RelationMetricV2:
    def __init__(self):
        self.p = []
        self.r = []
        self.f1 = []

    def update_state(self, batch_preds, batch_trues):        
        batch_preds = batch_preds.cpu().numpy()
        batch_trues = batch_trues.cpu().numpy()
        
        p = precision_score(batch_trues, batch_preds, average="macro")
        r = recall_score(batch_trues, batch_preds, average="macro")
        f1 = f1_score(batch_trues, batch_preds, average="macro")
            
        self.p.append(p)
        self.r.append(r)
        self.f1.append(f1)
    
    def result(self):
        return self.score()
    
    def reset(self):
        self.p = []
        self.r = []
        self.f1 = []
    
    def score(self):
        p = sum(self.p) / len(self.p)
        r = sum(self.r) / len(self.r)
        f1 = sum(self.f1) / len(self.f1)

        return p, r, f1

RelationMetric = RelationMetricV2
