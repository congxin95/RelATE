#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import git
import time
import sys
import os
import socket
from collections import OrderedDict

import pdb

class Logger:
    def __init__(self):
        self.current_time = self.get_current_time()
        self.commit_id = self.get_current_commit_id()
        self.gpu_id = self.get_gpu_id()
        self.delimiter = "&"
    
    def get_current_time(self):
        current_time = time.time()
        current_time = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)        
        return current_time
    
    def get_current_commit_id(self):
        repo = git.Repo()
        commit_id = repo.head.object.hexsha
        commit_id = commit_id[:7]        
        return commit_id
    
    def get_gpu_id(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        else:
            gpu_id = "cpu"
        return gpu_id
        
    def create_log(self, opt, dev_p, dev_r, dev_f1, P, R, F1):
        log_output = OrderedDict()
        log_output['model'] = opt.encoder + "-" + opt.model
        log_output['N'] = str(opt.evalN)
        log_output['K'] = str(opt.K)
        log_output['Dev P'] = str(dev_p)
        log_output['Dev R'] = str(dev_r)
        log_output['Dev F1'] = str(dev_f1)
        log_output['Test P'] = str(P)
        log_output['Test R'] = str(R)
        log_output['Test F1'] = str(F1)
        log_output['Tips'] = str(opt.notes)
        log_output['Save Checkpoint'] = str(opt.save_ckpt)
        log_output['Load Checkpoint'] = str(opt.load_ckpt)
        log_output['Commit ID'] = self.commit_id
        log_output["Date"] = self.current_time
        log_output['Seed'] = str(opt.seed)
        log_output['Device'] = ""
        log_output['CUDA'] = self.gpu_id
        log_output["Cmd"] = " ".join(sys.argv)
        log_output["Param"] = str(opt)
        
        log_output = [j for i, j in log_output.items()]
        log_output = self.delimiter.join(log_output)
        
        return log_output

if __name__ == "__main__":    
    import argparse
    import config
    
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
    
    parser.add_argument('--sample_num', default=config.sample_num, type=int, 
                        help='sample num of MC')
    parser.add_argument('--head_num', default=config.head_num, type=int, 
                        help='head num of proto interaction layer')
    parser.add_argument('--interact_layer_num', default=config.interact_layer_num, type=int, 
                        help='layer num of proto interaction layer')
    
    parser.add_argument('--encoder', default=config.encoder, type=str, 
                        help='bert')
    parser.add_argument('--feature_size', default=config.feature_size, type=int, 
                        help='feature size')
    parser.add_argument('--max_length', default=config.max_length, type=int, 
                        help='max sentence length')
    parser.add_argument('--encoder_path', default=config.encoder_path, type=str, 
                        help='pretrained encoder path')
        
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
    
    parser.add_argument('--device', default=config.device, type=str, 
                        help='device')
    parser.add_argument('--test', default=config.test, action="store_true",
                        help='test mode')
    
    parser.add_argument('--metric', default=config.metric, type=str,
                        help='evaluation metric')
    
    parser.add_argument('--notes', default=config.notes, type=str,
                        help='experiment notes')    
    
    opt = parser.parse_args()
    
    logger = Logger()
    print(logger.create_log(opt, 0, 0, 0, 1, 1, 1))