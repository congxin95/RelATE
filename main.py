#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import sys

import random
import numpy as np

import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW, get_linear_schedule_with_warmup

from model import RelATE

from model.encoder import BertEncoder
from dataloader import get_loader
from framework import Framework
from metric import TripletMetric, EntityMetric, HeadMetric, TailMetric, RelationMetric
import config
from utils import Logger

def main():
    # logger
    logger = Logger()
    
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=config.seed, type=int, 
                        help='seed')

    parser.add_argument('--train_set_path', default=config.train_set_path, type=str, 
                        help='train set path')
    parser.add_argument('--dev_set_path', default=config.dev_set_path, type=str, 
                        help='dev set path')
    parser.add_argument('--test_set_path', default=config.test_set_path, type=str, 
                        help='test set path')
    
    parser.add_argument('--model', default=config.model, type=str, 
                        help='model')
    
    parser.add_argument('--sent_sim', default=config.sent_sim, type=str, 
                        help='sent sim')
    parser.add_argument('--token_sim', default=config.token_sim, type=str, 
                        help='token sim')
    parser.add_argument('--pred_sent', default=config.pred_sent, action="store_true",
                        help='pred sent')
    
    parser.add_argument('--use_att_sent_emb', default=config.use_att_sent_emb, action="store_true",
                        help='use att sent emb')
    parser.add_argument('--use_auxiliary_loss', default=config.use_auxiliary_loss, action="store_true",
                        help='use auxiliary loss')
    parser.add_argument('--auxiliary_coef', default=config.auxiliary_coef, type=float, 
                        help='auxiliary coef')
    
    parser.add_argument('--encoder', default=config.encoder, type=str, 
                        help='bert')
    parser.add_argument('--feature_size', default=config.feature_size, type=int, 
                        help='feature size')
    parser.add_argument('--max_length', default=config.max_length, type=int, 
                        help='max sentence length')
    parser.add_argument('--encoder_path', default=config.encoder_path, type=str, 
                        help='pretrained encoder path')
    parser.add_argument('--gradient_checkpointing', default=config.gradient_checkpointing, action="store_true",
                        help='use gradient checkpointing for bert')
        
    parser.add_argument('--trainN', default=config.trainN, type=int, 
                        help='train N')
    parser.add_argument('--evalN', default=config.evalN, type=int, 
                        help='eval N')
    parser.add_argument('--K', default=config.K, type=int, 
                        help="K")
    parser.add_argument('--Q', default=config.Q, type=int, 
                        help="Q")
    
    parser.add_argument('--batch_size', default=config.batch_size, type=int, 
                        help='batch size')
    parser.add_argument('--num_workers', default=config.num_workers, type=int, 
                        help='number of worker in dataloader')

    parser.add_argument('--dropout', default=config.dropout, type=float, 
                        help='dropout rate')
    parser.add_argument('--optimizer', default=config.optimizer, type=str, 
                        help='sgd or adam or adamw')
    parser.add_argument('--learning_rate', default=config.learning_rate, type=float, 
                        help='learning rate for bert part')
    parser.add_argument('--learning_rate_2', default=config.learning_rate_2, type=float, 
                        help='learning rate for other part')
    parser.add_argument('--warmup_step', default=config.warmup_step, type=int, 
                        help='warmup step of bert')
    parser.add_argument('--scheduler_step', default=config.scheduler_step, type=int, 
                        help='scheduler step')
    parser.add_argument('--grad_clip_norm', default=config.grad_clip_norm, type=int, 
                        help='gradient clip norm')
    
    parser.add_argument('--train_epoch', default=config.train_epoch, type=int, 
                        help='train epoch')
    parser.add_argument('--eval_epoch', default=config.eval_epoch, type=int, 
                        help='eval epoch')
    parser.add_argument('--eval_step', default=config.eval_step, type=int, 
                        help='eval step')
    parser.add_argument('--test_epoch', default=config.test_epoch, type=int, 
                        help='test epoch')
    
    parser.add_argument('--ckpt_dir', default=config.ckpt_dir, type=str, 
                        help='checkpoint dir')
    parser.add_argument('--load_ckpt', default=config.load_ckpt, type=str, 
                        help='load checkpoint')
    parser.add_argument('--save_ckpt', default=config.save_ckpt, type=str, 
                        help='save checkpoint')
    
    parser.add_argument('--use_amp', default=config.use_amp, action="store_true",
                        help='use amp')
    parser.add_argument('--device', default=config.device, type=str, 
                        help='device')
    parser.add_argument('--test', default=config.test, action="store_true",
                        help='test mode')
    
    parser.add_argument('--metric', default=config.metric, type=str,
                        help='evaluation metric')
    
    parser.add_argument('--notes', default=config.notes, type=str,
                        help='experiment notes')    
    
    opt = parser.parse_args()
    
    # experiment notes
    print("Experiment notes :", opt.notes)
    
    # set seed
    if opt.seed is None:
        opt.seed = round((time.time() * 1e4) % 1e4)
    print(f"Seed: {opt.seed}")
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if not opt.test and opt.save_ckpt is None:
        opt.save_ckpt = os.path.join(opt.ckpt_dir, 
                                     "_".join([opt.model, 
                                               str(opt.evalN),
                                               str(opt.K),
                                               time.strftime('%Y%m%d_%H%M%S') + ".ckpt"]))
    print(f"Save checkpoint : {opt.save_ckpt}")
        
    if opt.load_ckpt is not None:
        print(f"Load checkpoint : {opt.load_ckpt}")
    
    if opt.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(opt.device)
    
    print("Hyperparameters :", opt)
    
    # define encoder
    encoder = BertEncoder(opt.encoder_path, opt.max_length, 
                          use_amp=opt.use_amp,
                          gradient_checkpointing=opt.gradient_checkpointing,)
    
    # load dataset    
    train_dataloader = get_loader(opt.train_set_path,
                                  opt.max_length, 
                                  encoder.tokenizer, 
                                  opt.trainN, opt.K, opt.Q,
                                  opt.batch_size)
    dev_dataloader = get_loader(opt.dev_set_path, 
                                opt.max_length, 
                                encoder.tokenizer, 
                                opt.evalN, opt.K, opt.Q,
                                opt.batch_size)
    test_dataloader = get_loader(opt.test_set_path, 
                                 opt.max_length, 
                                 encoder.tokenizer, 
                                 opt.evalN, opt.K, opt.Q,
                                 opt.batch_size)
    
    # define model
    
    if opt.model == "relate":
        model = RelATE(encoder, opt.feature_size, opt.max_length, opt.dropout, 
                        sent_sim=opt.sent_sim, 
                        token_sim=opt.token_sim,
                        pred_sent=opt.pred_sent,
                        use_att_sent_emb=opt.use_att_sent_emb,
                        use_auxiliary_loss=opt.use_auxiliary_loss,
                        auxiliary_coef=opt.auxiliary_coef)
    else:
        raise Exception("Invalid model!")
    model.to(device)
    
    # define optimizer and scheduler    
    if opt.optimizer == "adamw":
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [ 
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.01},                                                                                                                         
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0},
            {'params': [p for n, p in parameters_to_optimize
                if not 'bert' in n], 'lr': opt.learning_rate_2}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=opt.learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_step, num_training_steps=opt.train_epoch)
    elif opt.optimizer == "sgd":
        parameters_to_optimize = list(model.parameters())
        optimizer = SGD(parameters_to_optimize, lr=opt.learning_rate)
        scheduler = StepLR(optimizer, opt.scheduler_step)
    elif opt.optimizer == "adam":
        parameters_to_optimize = list(model.parameters())
        optimizer = Adam(parameters_to_optimize, lr=opt.learning_rate)
        scheduler = StepLR(optimizer, opt.scheduler_step)
    else:
        raise ValueError("Invalid optimizer")
    
    # define metric
    if opt.metric == "triplet":
        metric = TripletMetric()
    elif opt.metric == "entity":
        metric = EntityMetric()
    elif opt.metric == "head":
        metric = HeadMetric()
    elif opt.metric == "tail":
        metric = TailMetric()
    elif opt.metric == "relation":
        metric = RelationMetric()
    else:
        raise ValueError("Invalid metric")
    
    # define framework
    framework = Framework(train_dataloader=train_dataloader, 
                          dev_dataloader=dev_dataloader,
                          test_dataloader=test_dataloader,
                          metric=metric,
                          device=device,
                          opt=opt)
    # train
    if not opt.test:
        dev_p, dev_r, dev_f1 = framework.train(model,
                                               opt.trainN, opt.evalN, opt.K, opt.Q,
                                               optimizer,
                                               scheduler,
                                               opt.train_epoch,
                                               opt.eval_epoch,
                                               opt.eval_step,
                                               load_ckpt=opt.load_ckpt,
                                               save_ckpt=opt.save_ckpt,
                                               evaluate_relation=opt.pred_sent,
                                               use_amp=opt.use_amp,
                                               grad_clip_norm=opt.grad_clip_norm)
        checkpoint = opt.save_ckpt
    else:
        dev_p, dev_r, dev_f1 = 0, 0, 0
        checkpoint = opt.load_ckpt
    
    # test
    P, R, F1 = framework.evaluate(model, 
                                  opt.test_epoch, 
                                  opt.evalN, opt.K, opt.Q,
                                  mode="test",
                                  load_ckpt=checkpoint,
                                  evaluate_relation=opt.pred_sent,
                                  use_amp=opt.use_amp)
    print(f"Test result - P : {P:.6f}, R : {R:.6f}, F1 : {F1:.6f}")
    
    # finish
    print("Experiment notes :", opt.notes)
    
    print("Log output:")
    print(logger.create_log(opt, dev_p, dev_r, dev_f1, P, R, F1))
    

if __name__ == "__main__":
    main()
