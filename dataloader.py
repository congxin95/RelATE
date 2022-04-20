#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from collections import OrderedDict
from copy import deepcopy
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import pdb

INPUT_SCHEMA = {'tokens': [], 'token-label': [], 'sent-label': [], 'entity-mask': [], "att-mask": [], 'head-span': [], 'tail-span': [], "relation_id": [], "instance_id": []}

class FSJEDataset(Dataset):
    def __init__(self, 
                 dataset_path, 
                 max_length, 
                 tokenizer,
                 N, K, Q):
        self.raw_data = json.load(open(dataset_path, "r"))
        self.classes = self.raw_data.keys()
        
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.N = N
        self.K = K
        self.Q = Q
        
    def __len__(self):
        return 99999999
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        label2id, id2label = self.build_dict(target_classes)
        
        support_set = deepcopy(INPUT_SCHEMA)
        query_set = deepcopy(INPUT_SCHEMA)
        
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.raw_data[class_name]))), 
                    self.K + self.Q, False)
            class_label = torch.tensor(i)
            relation_id = torch.tensor(int(class_name[1:]))
            
            count = 0
            for j in indices:
                instance_id = torch.tensor(j)
                
                if count < self.K:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name, [class_name])
                    token_ids, label_ids, entity_mask, att_mask, head_span, tail_span = self.tokenize(instance, label2id)
                    
                    self.additem(support_set, 
                                 token_ids, label_ids, class_label, entity_mask, att_mask, head_span, tail_span,
                                 relation_id, instance_id)
                else:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name, target_classes)
                    token_ids, label_ids, entity_mask, att_mask, head_span, tail_span = self.tokenize(instance, label2id)
                    
                    self.additem(query_set, 
                                 token_ids, label_ids, class_label, entity_mask, att_mask, head_span, tail_span,
                                 relation_id, instance_id)
                count += 1
        
        for k, v in support_set.items():
            support_set[k] = torch.stack(v)
        
        for k, v in query_set.items():
            query_set[k] = torch.stack(v)
        
        return support_set, query_set, id2label
    
    def additem(self, 
                data, 
                token_ids, token_label_ids, sent_label_ids, entity_mask, att_mask, head_span, tail_span, 
                relation_id, instance_id):
        
        data['tokens'].append(token_ids)
        data['token-label'].append(token_label_ids)
        data['sent-label'].append(sent_label_ids)
        data['entity-mask'].append(entity_mask)
        data['att-mask'].append(att_mask)
        data['head-span'].append(head_span)
        data['tail-span'].append(tail_span)
        
        data['relation_id'].append(relation_id)
        data['instance_id'].append(instance_id)

    def preprocess(self, instance, relation, relation_list):
        result = {'tokens': [], 'token-label': [], 'entity-mask': [], 'head-span': [], 'tail-span': []}

        tokens = instance['tokens']
        label = ['O'] * len(tokens)
        entity_mask = [0] * len(tokens)
        
        head_start = [0] * len(tokens)
        head_end = [0] * len(tokens)
        tail_start = [0] * len(tokens)
        tail_end = [0] * len(tokens)
        
        head_pos = instance['h'][2][0]
        tail_pos = instance['t'][2][0]        
        
        for i, idx in enumerate(head_pos):
            if i == 0:
                label[idx] = f"B-{relation}:HEAD"
                entity_mask[idx] = 1
            else:
                label[idx] = f"I-{relation}:HEAD"
                entity_mask[idx] = 2
        
        for i, idx in enumerate(tail_pos):
            if i == 0:
                label[idx] = f"B-{relation}:TAIL"
                entity_mask[idx] = 3
            else:
                label[idx] = f"I-{relation}:TAIL"
                entity_mask[idx] = 4
        
        head_start[head_pos[0]] = 1
        head_end[head_pos[-1]] = 1
        tail_start[tail_pos[0]] = 1
        tail_end[tail_pos[-1]] = 1
        
        result['tokens'] = tokens
        result['token-label'] = label
        result['entity-mask'] = entity_mask
        result['head-span'] = [head_start, head_end]
        result['tail-span'] = [tail_start, tail_end]
        
        return result
    
    def build_dict(self, relation_list):
        label2id = OrderedDict()
        label2id['O'] = 0
        
        for i, relation in enumerate(relation_list):
            label2id['B-' + relation + ":HEAD"] = 4*i + 1
            label2id['I-' + relation + ":HEAD"] = 4*i + 2
            label2id['B-' + relation + ":TAIL"] = 4*i + 3
            label2id['I-' + relation + ":TAIL"] = 4*i + 4
        
        id2label = OrderedDict({j: i for i, j in label2id.items()})
        
        return label2id, id2label

    def tokenize(self, instance, label2id):
        max_length = self.max_length
        
        raw_tokens = instance['tokens']
        raw_label = instance['token-label']
        raw_entity_mask = instance['entity-mask']
        raw_head_start, raw_head_end = instance['head-span']
        raw_tail_start, raw_tail_end = instance['tail-span']
        
        # token -> index
        tokens = ['[CLS]']
        label = ['O']
        entity_mask = [0]
        head_start = [0]
        head_end = [0]
        tail_start = [0]
        tail_end = [0]
        for i, token in enumerate(raw_tokens):
            tokenize_result = self.tokenizer.tokenize(token)
            tokens += tokenize_result
            
            if len(tokenize_result) > 1:
                label += [raw_label[i]]
                entity_mask += [raw_entity_mask[i]]
                
                if raw_label[i][0] == "B":
                    tmp_label = "I" + raw_label[i][1:]
                    label += [tmp_label] * (len(tokenize_result) - 1)
                    entity_mask += [raw_entity_mask[i] + 1] * (len(tokenize_result) - 1)
                else:
                    label += [raw_label[i]] * (len(tokenize_result) - 1)
                    entity_mask += [raw_entity_mask[i]] * (len(tokenize_result) - 1)
                
                if raw_head_start[i] == 1:
                    head_start += [1] + [0] * (len(tokenize_result) - 1)
                else:
                    head_start += [raw_head_start[i]] * (len(tokenize_result))
                
                if raw_head_end[i] == 1:
                    head_end += [0] * (len(tokenize_result) - 1) + [1]
                else:
                    head_end += [raw_head_end[i]] * (len(tokenize_result))
                
                if raw_tail_start[i] == 1:
                    tail_start += [1] + [0] * (len(tokenize_result) - 1)
                else:
                    tail_start += [raw_tail_start[i]] * (len(tokenize_result))
                
                if raw_tail_end[i] == 1:
                    tail_end += [0] * (len(tokenize_result) - 1) + [1]
                else:
                    tail_end += [raw_tail_end[i]] * (len(tokenize_result))
                
            else:
                label += [raw_label[i]] * len(tokenize_result)
                entity_mask += [raw_entity_mask[i]] * len(tokenize_result)
                
                head_start += [raw_head_start[i]] * len(tokenize_result)
                head_end += [raw_head_end[i]] * len(tokenize_result)
                tail_start += [raw_tail_start[i]] * len(tokenize_result)
                tail_end += [raw_tail_end[i]] * len(tokenize_result)
        
        # add SEP
        tokens += ['[SEP]']
        label += ['O']
        entity_mask += [0]
        head_start += [0]
        head_end += [0]
        tail_start += [0]
        tail_end += [0]
        
        # att mask
        att_mask = torch.zeros(max_length)
        att_mask[:len(tokens)] = 1
        
        # padding
        while len(tokens) < self.max_length:
            tokens.append('[PAD]')
            label.append('O')
            entity_mask.append(0)
            head_start.append(0)
            head_end.append(0)
            tail_start.append(0)
            tail_end.append(0)
        tokens = tokens[:max_length]
        label = label[:max_length]
        entity_mask = entity_mask[:max_length]
        head_start = head_start[:max_length]
        head_end = head_end[:max_length]
        tail_start = tail_start[:max_length]
        tail_end = tail_end[:max_length]
        
        # to ids
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor(token_ids).long()
        
        label_ids = list(map(lambda x: label2id[x], label))
        label_ids = torch.tensor(label_ids).long()
        
        entity_mask_ids = torch.tensor(entity_mask)
        
        head_span = torch.tensor([head_start, head_end]).long()
        tail_span = torch.tensor([tail_start, tail_end]).long()
        
        return token_ids, label_ids, entity_mask_ids, att_mask, head_span, tail_span


def collate_fn(data):
    batch_support = deepcopy(INPUT_SCHEMA)
    batch_query = deepcopy(INPUT_SCHEMA)
    batch_id2label = []
    
    support_sets, query_sets, id2labels = zip(*data)
    
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k].append(support_sets[i][k])
        for k in query_sets[i]:
            batch_query[k].append(query_sets[i][k])
        batch_id2label.append(id2labels[i])
    
    for k in batch_support:
        batch_support[k] = torch.cat(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.cat(batch_query[k], 0)
    
    return batch_support, batch_query, batch_id2label

def get_loader(dataset_path,
               max_length, 
               tokenizer,
               N, K, Q,
               batch_size, 
               num_workers=8, 
               collate_fn=collate_fn):
    
    dataset = FSJEDataset(dataset_path, 
                          max_length, 
                          tokenizer,
                          N, K, Q)
        
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return iter(dataloader)
