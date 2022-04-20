# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers import BertTokenizer, BertModel

class BertEncoder(nn.Module):
    def __init__(self, bert_path, max_length, use_amp=False, **config):
        super(BertEncoder, self).__init__()
        
        self.max_length = max_length
        self.use_amp = use_amp
        
        self.bert = BertModel.from_pretrained(bert_path, **config)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
    
    def forward(self, input_ids, **bert_args):
        with autocast(enabled=self.use_amp):
            output = self.bert(input_ids, **bert_args)
        
        output = output[0]
        return output

