#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad

import json
import pdb

class RelATE(nn.Module):
    def __init__(self, 
                 encoder, 
                 feature_size, 
                 max_len, 
                 dropout_rate,
                 sent_sim="conv",
                 token_sim="conv",
                 pred_sent=False,
                 use_att_sent_emb=False,
                 use_auxiliary_loss=False,
                 auxiliary_coef=1.0):
        super(RelATE, self).__init__()
        
        self.tag_num = 4
        self.feature_size = feature_size
        self.max_len = max_len
        
        self.encoder = encoder
        self.encoder = nn.DataParallel(self.encoder)
        
        self.cost = nn.CrossEntropyLoss(reduction="none")
        
        self.dropout_rate = dropout_rate
        self.drop = nn.Dropout(self.dropout_rate)
        
        self.sent_sim = sent_sim
        self.token_sim = token_sim
        
        self.pred_sent = pred_sent
        
        self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=5,
                    out_channels=8,
                    kernel_size=3,
                    padding=1
                    ),
                nn.ReLU(),
                nn.MaxPool1d(
                    kernel_size=2
                    ),
                nn.Dropout(self.dropout_rate),
                
                nn.Conv1d(
                    in_channels=8,
                    out_channels=4,
                    kernel_size=3,
                    padding=1
                    ),
                nn.ReLU(),
                nn.MaxPool1d(
                    kernel_size=2
                    ),
                nn.Dropout(self.dropout_rate)
        )
        self.fc = nn.Linear(768, 1)
        self.token_cost = nn.BCEWithLogitsLoss(reduction="none")
        
        self.use_att_sent_emb = use_att_sent_emb

        self.use_auxiliary_loss = use_auxiliary_loss
        self.auxiliary_coef = auxiliary_coef
        self.W_auxiliary = nn.Linear(2*self.feature_size, self.feature_size)
        
        self.mse_cost = nn.MSELoss()
        
    def forward(self, support_set, query_set, N, K, Q):
        # encode
        support_emb = self.encode(support_set)   # B*N*K, max_len, feature_size
        query_emb = self.encode(query_set)       # B*N*Q, max_len, feature_size

        support_emb = support_emb.view(-1, N, K, self.max_len, self.feature_size)   # B, N, K, max_len, feature_size
        query_emb = query_emb.view(-1, N*Q, self.max_len, self.feature_size)        # B, N*Q, max_len, feature_size

        support_entity_mask = support_set["entity-mask"].view(-1, N, K, self.max_len)
        query_entity_mask = query_set["entity-mask"].view(-1, N*Q, self.max_len)
        
        support_att_mask = support_set["att-mask"].view(-1, N, K, self.max_len)
        query_att_mask = query_set["att-mask"].view(-1, N*Q, self.max_len)
        
        support_head_span = support_set["head-span"].view(-1, N, K, 2, self.max_len)    # B, N, K, 2, max_len
        support_tail_span = support_set["tail-span"].view(-1, N, K, 2, self.max_len)    # B, N, K, 2, max_len
        query_head_span = query_set["head-span"].view(-1, N*Q, 2, self.max_len)         # B, NQ, 2, max_len
        query_tail_span = query_set["tail-span"].view(-1, N*Q, 2, self.max_len)         # B, NQ, 2, max_len
        
        # sent cls
        support_sent = self.sent_embed(support_emb, 
                                       entity_mask=support_entity_mask,
                                       att_mask=support_att_mask)       # B, N, K, feature_size
        sent_prototype = self.sent_proto(support_sent)                  # B, N, feature_size
        
        query_sent = self.sent_embed(query_emb,
                                     sent_prototype=sent_prototype,
                                     att_mask=query_att_mask)           # B, NQ, feature_size
        sent_logits = self.sent_similarity(sent_prototype, query_sent)  # B, NQ, N
        
        sent_label = query_set['sent-label'].view(-1, N*Q)              # B, NQ
        sent_loss = self.loss(sent_logits, sent_label)
        _, sent_pred = torch.max(sent_logits.view(-1, sent_logits.shape[-1]), 1)    # B*N*Q*max_len
        
        # condition
        if self.training:
            query_condition = sent_label
        else:
            query_condition = sent_pred.view(-1, N*Q)
        
        # token cls
        token_loss = []
        token_pred = []
        auxiliary_loss = []
        for k in range(query_emb.shape[0]): # batch num
            for i in range(query_emb.shape[1]): # query num
                j = query_condition[k][i]
                
                support = support_emb[k:k+1, j:j+1]                     # 1, 1, K, max_len, feature_size
                support_token_label = support_entity_mask[k:k+1, j:j+1] # 1, 1, K, max_len
                support_token_att_mask = support_att_mask[k:k+1, j:j+1] # 1, 1, K, max_len
                support_token_head_span = support_head_span[k:k+1, j:j+1] # 1, 1, K, 2, max_len
                support_token_tail_span = support_tail_span[k:k+1, j:j+1] # 1, 1, K, 2, max_len
                
                if query_emb.dim() > 4:
                    query = query_emb[k:k+1, i:i+1, j]                  # 1, 1, max_len, feature_size
                else:
                    query = query_emb[k:k+1, i:i+1]                     # 1, 1, max_len, feature_size
                token_label = query_entity_mask[k:k+1, i:i+1]           # 1, 1, max_len
                token_att_mask = query_att_mask[k:k+1, i:i+1]           # 1, 1, max_len
                token_head_span = query_head_span[k:k+1, j:j+1]         # 1, 1, 2, max_len
                token_tail_span = query_tail_span[k:k+1, j:j+1]         # 1, 1, 2, max_len
                
                token_prototype = self.token_proto(support, 
                                                   support_token_label, 
                                                   support_token_att_mask,
                                                   support_token_head_span,
                                                   support_token_tail_span)  # B, tag_num, feature_size
                
                token_logits = self.token_similarity(token_prototype, query) # B, NQ, max_len, tag_num
                
                # softmax
                t_token_loss = self.loss(token_logits, 
                                         token_label, 
                                         token_head_span, 
                                         token_tail_span,
                                         att_mask=token_att_mask.view(-1))
                
                # prediction
                t_token_logits = token_logits - 1000*(1 - token_att_mask.unsqueeze(-1)).to(token_logits)    # 1, 1, max_len, tag_num
                _, t_token_pred_idx = t_token_logits.max(dim=-2)                # 1, 1, tag_num
                
                t_token_pred = torch.zeros_like(token_label)                    # 1, 1, max_len
                t_token_pred[0, 0, t_token_pred_idx[0, 0, 0]] = 1
                t_token_pred[0, 0, t_token_pred_idx[0, 0, 0]+1:t_token_pred_idx[0, 0, 1]+1] = 2
                t_token_pred[0, 0, t_token_pred_idx[0, 0, 2]] = 3
                t_token_pred[0, 0, t_token_pred_idx[0, 0, 2]+1:t_token_pred_idx[0, 0, 3]+1] = 4
                
                if self.use_auxiliary_loss:
                    t_head_pos = ((support_token_label == 1) | (support_token_label == 2)).unsqueeze(-1).float()  # 1, 1, K, max_len, 1
                    t_tail_pos = ((support_token_label == 3) | (support_token_label == 4)).unsqueeze(-1).float()  # 1, 1, K, max_len, 1
                    t_head_emb = (support * t_head_pos).sum(-2) / t_head_pos.sum(-2)            # 1, 1, K, feature_size
                    t_tail_emb = (support * t_tail_pos).sum(-2) / t_tail_pos.sum(-2)            # 1, 1, K, feature_size
                    t_head_emb = t_head_emb.mean(-2)                                            # 1, 1, feature_size
                    t_tail_emb = t_tail_emb.mean(-2)                                            # 1, 1, feature_size
                    t_auxiliary_emb = self.W_auxiliary(torch.cat([t_head_emb, t_tail_emb], dim=-1)) # 1, 1, feature_size
                    t_sent_prototype = sent_prototype[k:k+1, j:j+1] # 1, 1, feature_size
                    t_auxiliary_loss = self.mse_cost(t_sent_prototype.view(-1, self.feature_size), 
                                                     t_auxiliary_emb.view(-1, self.feature_size))
                    auxiliary_loss.append(t_auxiliary_loss)
                
                token_pred.append(t_token_pred)
                token_loss.append(t_token_loss)
        token_loss = sum(token_loss) / len(token_loss)
        
        # final loss
        loss = sent_loss + token_loss
        
        if self.use_auxiliary_loss:
            auxiliary_loss = sum(auxiliary_loss) / len(auxiliary_loss)
            loss = loss + self.auxiliary_coef * auxiliary_loss
        
        # final pred
        if self.pred_sent:
            pred = sent_pred
        else:
            for i, (s, t) in enumerate(zip(sent_pred, token_pred)):
                mask = (t != 0)
                token_pred[i] = (4*s + t) * mask.long()
            pred = torch.stack(token_pred)
        
        outputs = (loss, None, pred)
        
        return outputs
    
    def encode(self, inputs):
        inputs_emb = self.encoder(inputs['tokens'], attention_mask=inputs["att-mask"])
        inputs_emb = self.drop(inputs_emb)
        return inputs_emb
    
    def sent_embed(self, token_emb, entity_mask=None, sent_prototype=None, att_mask=None):
        if self.use_att_sent_emb:
            if entity_mask is not None:
                head_pos = ((entity_mask == 1) | (entity_mask == 2)).unsqueeze(-1).float()  # B, N, K, max_len, 1
                tail_pos = ((entity_mask == 3) | (entity_mask == 4)).unsqueeze(-1).float()  # B, N, K, max_len, 1
                
                head_emb = (token_emb * head_pos).sum(-2) / head_pos.sum(-2)            # B, N, K, feature_size
                tail_emb = (token_emb * tail_pos).sum(-2) / tail_pos.sum(-2)            # B, N, K, feature_size
                cls_emb = token_emb[:, :, :, 0, :]                                      # B, N, K, feature_size
                
                rel_emb = cls_emb + head_emb + tail_emb                                 # B, N, K, feature_size
                rel_emb = rel_emb.unsqueeze(-1)                                         # B, N, K, feature_size, 1
                
                att_mask = att_mask.unsqueeze(-1)                                       # B, N, K, max_len, 1
                
                att_coef = torch.matmul(token_emb, rel_emb)                             # B, N, K, max_len, 1
                att_coef = att_coef - 1e3 * (1 - att_mask)                              # B, N, K, max_len, 1
                att_coef = att_coef.softmax(-2)                                         # B, N, K, max_len, 1
                
                sent_emb = (att_coef * token_emb).sum(-2)                               # B, N, K, feature_size
            elif sent_prototype is not None:
                N = sent_prototype.shape[1]
                NQ = token_emb.shape[1]
                
                sent_prototype = sent_prototype.unsqueeze(1)                            # B, 1, N, feature_size
                sent_prototype = sent_prototype.expand(-1, NQ, -1, -1)                  # B, NQ, N, feature_size
                
                # sent_prototype = self.Wq_proto(sent_prototype)                          # B, NQ, N, feature_size
                # token_emb = self.Wk_t(token_emb)                                        # B, NQ, max_len, feature_size
                
                att_mask = att_mask.unsqueeze(2)                                        # B, NQ, 1, max_len
                
                att_coef = torch.matmul(sent_prototype, token_emb.transpose(-1, -2))    # B, NQ, N, max_len
                att_coef = att_coef - 1e3 * (1 - att_mask)                              # B, NQ, N, max_len
                att_coef = att_coef.softmax(-1)                                         # B, NQ, N, max_len
                att_coef = att_coef.unsqueeze(-1)                                       # B, NQ, N, max_len, 1
                
                tmp_token_emb = token_emb.unsqueeze(-3)                                 # B, NQ, 1, max_len, feature_size
                tmp_token_emb = tmp_token_emb.expand(-1, -1, N, -1, -1)                 # B, NQ, N, max_len, feature_size
                
                sent_emb = (att_coef * tmp_token_emb).sum(-2)                           # B, NQ, N, feature_size
        else:
            original_shape = token_emb.shape
            token_emb = token_emb.view(-1, self.max_len, self.feature_size)
            
            sent_emb = token_emb[:, 0, :]
            sent_emb = sent_emb.view(*original_shape[:-2], self.feature_size)
        
        return sent_emb

    def sent_proto(self, support_sent):
        sent_prototype = support_sent.mean(-2)
        return sent_prototype
    
    def sent_similarity(self, prototype, query):
        '''
        :param: prototype: B, N feature_size
        :param: query: B, NQ, feature_size
        :return: similarity: B, NQ, N
        '''
        prototype = prototype.unsqueeze(1)  # B, 1, N, feature_size
        if query.dim() != 4:
            query = query.unsqueeze(2)          # B, NQ, 1, feature_size
        
        prototype, query = torch.broadcast_tensors(prototype, query) # B, NQ, N, feature_size
        
        if self.sent_sim == "euc":
            # euclidean distance
            similarity = -(torch.pow(prototype - query, 2)).sum(-1)
        elif self.sent_sim == "dot":
            # dot
            similarity = (prototype * query).sum(-1)
        elif self.sent_sim == "norm":
            # normalize
            prototype_norm = prototype.norm(dim=-1) + 1e-8                      # B, N*Q, max_len, tag_num
            similarity = (prototype * query).sum(-1) / prototype_norm           # B, N*Q, max_len, tag_num
            similarity -= 0.5 * prototype_norm                                  # B, N*Q, max_len, tag_num
        elif self.sent_sim == "conv":
            # conv relnet
            minus = (query - prototype).abs()                                   # B, NQ, N, feature_size
            add = query + prototype                                             # B, NQ, N, feature_size
            mul = query * prototype                                             # B, NQ, N, feature_size
            
            inputs = torch.stack([prototype, query, minus, add, mul], dim=-2)   # B, NQ, N, 5, feature_size
            original_shape = inputs.shape
            inputs = inputs.view(-1, 5, self.feature_size)                      # B*NQ*N, 5, feature_size
            out = self.conv(inputs)                                             # B*NQ*N, 4, 192
            out = out.view(*original_shape[:3], -1)                             # B, NQ, N, 768
            similarity = self.fc(out).squeeze(-1)                               # B, NQ, N
        else:
            raise ValueError("Invalid sent sim")
        
        return similarity

    def token_proto(self, support_emb, support_label, att_mask, head_span, tail_span):
        '''
        :param: 
            support_emb: B, N, K, max_len, feature_size
            support_label: B, N, K, max_len
            att_mask: B, N, K, max_len
            head_span: B, N, K, 2, max_len
            tail_span: B, N, K, 2, max_len
        :return: 
            token_proto: B, tag_num, feature_size
        '''
        B, N, K, max_len, feature_size = support_emb.shape
        prototype = torch.empty(B, self.tag_num, feature_size).to(support_emb)
        
        point_label = torch.cat([head_span, tail_span], dim=-2) # B, N, K, 4, max_len
        point_label = point_label.unsqueeze(-1)                 # B, N, K, 4, max_len, 1
        point_label = point_label.expand(-1, -1, -1, -1, -1, self.feature_size) # B, N, K, 4, max_len, feature_size
        point_label = point_label.to(support_emb)               # B, N, K, 4, max_len, feature_size
        
        att_mask = att_mask.unsqueeze(-1)

        for i in range(B):
            for j in range(self.tag_num):
                current_mask = (point_label[i, :, :, j] == 1).to(point_label)   # N, K, max_len, feature_size
                current_mask *= att_mask[i]                                     # N, K, max_len, feature_size
                
                sum_feature = (support_emb[i] * current_mask).reshape(-1, feature_size).sum(0)
                num_feature = current_mask.sum() / feature_size + 1e-8
                prototype[i, j] = sum_feature / num_feature
                
        return prototype

    def token_similarity(self, prototype, query):
        '''
        inputs:
            prototype: B, tag_num, feature_size
            query: B, NQ, max_len, feature_size
        outputs:
            sim: B, NQ, max_len, tag_num
        '''
        
        query = query.unsqueeze(-2)                                     # B, NQ, max_len, 1, feature_size
        prototype = prototype.unsqueeze(1).unsqueeze(2)                 # B, 1, 1, tag_num, feature_size
        prototype, query = torch.broadcast_tensors(prototype, query)    # B, NQ, max_len, tag_num, feature_size
        
        if self.token_sim == "conv":
            minus = (query - prototype).abs()
            add = query + prototype
            mul = query * prototype
            
            inputs = torch.stack([prototype, query, minus, add, mul], dim=-2)   # B, NQ, max_len, tag_num, 5, feature_size
            original_shape = inputs.shape
            inputs = inputs.view(-1, 5, self.feature_size)                      # B*NQ*max_len*tag_num, 5, feature_size
            out = self.conv(inputs)                                             # B*NQ*max_len*tag_num, 4, 192
            out = out.view(*original_shape[:4], -1)                             # B, NQ, max_len, tag_num, 768
            sim = self.fc(out).squeeze(-1)                                      # B, NQ, max_len, tag_num
            
        elif self.token_sim == "euc":
            sim = -(torch.pow(prototype - query, 2)).sum(-1)
        elif self.token_sim == "dot":
            sim = (prototype * query).sum(-1)
        elif self.token_sim == "norm":
            prototype_norm = prototype.norm(dim=-1) + 1e-8                      # B, N*Q, max_len, tag_num
            sim = (prototype * query).sum(-1) / prototype_norm           # B, N*Q, max_len, tag_num
            sim -= 0.5 * prototype_norm                                  # B, N*Q, max_len, tag_num
            
        return sim
    
    def loss(self, logits, label, head_span=None, tail_span=None, att_mask=None):
        logits = logits.view(-1, logits.shape[-1])
        
        if head_span is not None and tail_span is not None:
            golden = torch.cat([head_span, tail_span], dim=-2).transpose(-1, -2)    # B, NQ, max_len, tag_num
            golden = golden.view(-1, golden.shape[-1]).float()
            loss = self.token_cost(logits, golden)
        else:
            label = label.view(-1)
            loss = self.cost(logits, label)
        
        if att_mask is None:
            loss_weight = torch.ones_like(label).float()
            loss = (loss_weight * loss).mean()
        else:
            if att_mask.dim() != loss.dim():
                att_mask = att_mask.unsqueeze(-1)
            loss = (att_mask * loss).sum() / att_mask.sum()
            
        return loss
