#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
import os
import numpy as np
import json

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus, split_labels
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import onmt
from transformers import GPT2Tokenizer
import torch
from sklearn.metrics import accuracy_score
import pickle
def constraint_iter_func(f_iter):
    all_tags = []
    for json_line in f_iter:
        data = json.loads(json_line)
        words = data['words']
        probs = [p[1] for p in data['class_probabilities'][:len(words)]]
        tags = [1 if p > opt.bu_threshold else 0 for p in probs]
        all_tags.append(tags)
        #print(len(words), len(data['class_probabilities']))
        #all_tags.append(words)
    return all_tags


def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)
    load_test_model = onmt.model_builder.load_test_model
    _, model, _ = load_test_model(opt)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.decoder.get(50163)
    input_data = open(opt.src,'r').readlines()
    input_data = [i.strip() for i in input_data][:-100]
    labels = [0] * len(input_data)
    model = model.cuda()
    model.eval()
    predict_list = []
    with torch.no_grad():
        for i in range(0, len(input_data), 30):
            batch = input_data[i:i+30]
            tokens_ids = [tokenizer.encode(i, pad_to_max_length=True, max_length=200) for i in batch]
            lengths = torch.tensor([len(tokenizer.encode(i)) if len(tokenizer.encode(i)) < 200 else 200 for i in batch])
            tokens_ids = torch.tensor(tokens_ids)
            tokens_ids = torch.transpose(tokens_ids, 1, 0)
            tokens_ids.unsqueeze_(-1)
            tokens_ids = tokens_ids.cuda()
            lengths = lengths.cuda()
            kwargs = {"facts": None}
            result = model(tokens_ids, tokens_ids, lengths, **kwargs)[-1]
            result = result.tolist()
            predict_list.extend(result)
        print("Acc Score is {}".format(accuracy_score(y_true=labels, y_pred=predict_list)))


            


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser

def build_opt():
    with open("opt.pkl", 'rb') as f1:
        opt1 = pickle.load(f1)
    return opt1

if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    
    model_path = opt.models[0]
    step = os.path.basename(model_path)[:-3].split('step_')[-1]
    temp = opt.random_sampling_temp

    if opt.extra_output_str:
        opt.extra_output_str = '_'+opt.extra_output_str

    if opt.output is None:
        output_path = '/'.join(model_path.split('/')[:-2])+'/output_%s_%s%s.encoded' % (step, temp, opt.extra_output_str)
        opt.output = output_path
    print(opt.output)

    import pickle
    with open("opt.pkl", 'wb') as f1:
        pickle.dump(opt, f1)
    # exit()
    main(opt)
    
