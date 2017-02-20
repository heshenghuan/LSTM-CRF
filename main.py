#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-20 16:28:53

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import sys
import argparse
import functools
import numpy as np
import codecs as cs
from keras_src import lstm_ner


def add_arg(name, default=None, **kwarg):
    assert hasattr(
        add_arg, 'arg_parser'), "You must register an arg_parser with add_arg before calling it"
    if 'action' in kwarg:
        add_arg.arg_parser.add_argument(name, default=default, **kwarg)
    else:
        add_arg.arg_parser.add_argument(
            name, default=default, type=type(default), **kwarg)
    return


def add_arg_to_L(L, name, default=None, **kwarg):
    L.append(name[2:])
    add_arg(name, default, **kwarg)
    return

if __name__ == "__main__":
    ##########################################################################
    # PARSE ARGUMENTS, BUILD CIRCUIT, TRAIN, TEST
    ##########################################################################
    _arg_parser = argparse.ArgumentParser(description='LSTM')
    add_arg.arg_parser = _arg_parser
    TOPO_PARAM = []
    TRAIN_PARAM = []
    # File IO
    add_arg('--only_test', False, help='Only do the test')
    add_arg('--save_model_param', 'best-parameters',
            help='The best model will be saved there')
    add_arg('--training_data', r'../data/weiboNER.conll.train',
            help='training file name')
    add_arg('--valid_data', r'../data/weiboNER.conll.valid',
            help='validation file name')
    add_arg('--test_data', r'../data/weiboNER.conll.test',
            help='test file name')
    add_arg('--output_dir', '/export/projects/npeng/weiboNER_data/',
            help='the output dir that stores the prediction')
    add_arg('--eval_test', False,
            help='Whether evaluate the test data: test data may not have annotations.')
    add_arg('--emb_type', 'char',
            help='The embedding type, choose from (char, charpos)')
    add_arg('--emb_file', r'../embeddings/weibo_char_vectors',
            help='The initial embedding file name')
    add_arg('--ner_feature_thresh', 0,
            help="The minimum count OOV threshold for NER")
    # Training
    add_arg_to_L(TRAIN_PARAM, '--lr', 0.05)
    # add_arg_to_L(TRAIN_PARAM, '--use_emb', 'true')
    add_arg_to_L(TRAIN_PARAM, '--fine_tuning', 'true')
    add_arg_to_L(TRAIN_PARAM, '--nepochs', 200)
    add_arg_to_L(TRAIN_PARAM, '--neval_epochs', 5)
    add_arg_to_L(TRAIN_PARAM, '--optimizer', 'sgd')
    # add_arg_to_L(TRAIN_PARAM, '--seed', 1337, help='set random seed')
    add_arg_to_L(TRAIN_PARAM, '--decay', True,  action='store_true')
    add_arg_to_L(TRAIN_PARAM, '--decay_epochs', 10)
    add_arg_to_L(TRAIN_PARAM, '--minimum_lr', 1e-5)
    # Topology
#     add_arg_to_L(TOPO_PARAM, '--circuit', 'plainOrderOneCRF',
#                  help="the conbination of different models")
#     add_arg_to_L(TOPO_PARAM, '--in_dim',                       -1)
#     add_arg_to_L(TOPO_PARAM, '--emission_trans_out_dim', -1)
#     add_arg_to_L(TOPO_PARAM, '--L2Reg_reg_weight',             0.0)
#     add_arg_to_L(TOPO_PARAM, '--win',                          1)
    # DEBUG
    add_arg('--verbose', 2)
    args = _arg_parser.parse_args()
    lstm_ner.main(args)
    # from sighan_ner import loaddata, get_data, eval_ner, error_analysis  # , conlleval
    # main(args)
