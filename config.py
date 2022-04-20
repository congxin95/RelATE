#!/usr/bin/env python3
# -*- coding: utf-8 -*-

seed = None

train_set_path = "./data/train.json"
dev_set_path = "./data/dev.json"
test_set_path = "./data/test.json"
encoder_path = 'bert-base-uncased'

# model settings 
model = "relate"

sent_sim = "conv"
token_sim = "conv"
pred_sent = False

use_att_sent_emb = True
use_auxiliary_loss = True
auxiliary_coef = 0.2

# encoder settings
encoder = "bert"
feature_size = 768
max_length = 128
gradient_checkpointing = False

trainN = 5
evalN = 5
K = 5
Q = 1

batch_size = 1
num_workers = 8

dropout = 0.1
optimizer = "adamw"
learning_rate = 1e-5
learning_rate_2 = 1e-4
warmup_step = 100
scheduler_step = 1000
grad_clip_norm = 10

train_epoch = 30000
eval_epoch = 1000
eval_step = 500
test_epoch = 3000

ckpt_dir = "checkpoint/"
load_ckpt = None
save_ckpt = None

use_amp = False
device = None
test=False

metric = "triplet"

notes=""
