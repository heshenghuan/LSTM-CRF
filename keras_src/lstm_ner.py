#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-20 16:47:02

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import sys
import keras
import argparse
import functools
import numpy as np
import codecs as cs
from train_util import load_params, get_data, conv_data
from evaluate_util import eval_ner
from constant import MAX_LEN


def convert_id_to_word(corpus, idx2label):
    return [[idx2label[word] for word in sentence]
            for sentence
            in corpus]


def predict(features, words, idx2label, idx2word,
            _args, f_classify, groundtruth=None):
    predictions = convert_id_to_word(
        # _conv_x(sentence, _args.win, _args.vocsize))
        [f_classify(f, w) for f, w in zip(features, words)],
        idx2label
    )
    if groundtruth is None:
        return None, predictions
    # conlleval(predictions, groundtruth,
    results = eval_ner(predictions, groundtruth)
    #          folder + '/current.valid.txt', folder)
    # error_analysis(words, predictions, groundtruth, idx2word)
    return results, predictions


def write_prediction(filename, output_dir, lex_test, pred_test):
    f_path = os.path.join(os.path.basename(output_dir), filename)
    with cs.open(f_path, 'w', encoding='utf-8') as outf:
        for sent_w, sent_l in zip(lex_test, pred_test):
            assert len(sent_w) == len(sent_l)
            for w, l in zip(sent_w, sent_l):
                outf.write(w + '\t' + l + '\n')
            outf.write('\n')


def main(_args):
    if _args.only_test:
        cargs = {}
        print "loading parameters!"
        load_params(_args.save_model_param, cargs)
        test_feat, test_lex_orig, test_y = get_data(
            _args.test_data, cargs['feature2idx'], cargs['word2idx'],
            cargs['label2idx'], cargs['emb_type'],
            anno=None, has_label=_args.eval_test)

        test_feat, test_lex, test_y = conv_data(
            test_feat, test_lex_orig, test_y, cargs['win'], cargs['vocsize'])
        idx2label = dict((k, v) for v, k in cargs['label2idx'].iteritems())
        idx2word = dict((k, v) for v, k in cargs['word2idx'].iteritems())
        groundtruth_test = None
        if _args.eval_test:
            groundtruth_test = convert_id_to_word(test_y, idx2label)
        original_text = convert_id_to_word(test_lex_orig, idx2word)
        f_classify = cargs['f_classify']
        res_test, pred_test = predict(
            test_feat, test_lex, idx2label, idx2word,
            _args, f_classify, groundtruth_test)
        write_prediction(_args.test_data + '.prediction',
                         _args.output_dir, original_text, pred_test)
        exit(0)

    print "loading data from:", _args.training_data, _args.valid_data, _args.test_data
    train_set, valid_set, test_set, dicts = loaddata(
        _args.training_data, _args.valid_data, _args.test_data, feature_thresh=_args.ner_feature_thresh, mode=_args.emb_type, test_label=_args.eval_test)
    train_feat, train_lex_orig, train_y = train_set
    valid_feat, valid_lex_orig, valid_y = valid_set
    test_feat, test_lex_orig, test_y = test_set
    feature2idx = dicts['features2idx']
    word2idx = dicts['words2idx']
    label2idx = dicts['labels2idx']
    # idx2feature = dict((k, v) for v, k in feature2idx.iteritems())
    _args.label2idx = label2idx
    _args.word2idx = word2idx
    _args.feature2idx = feature2idx
    nclasses = len(label2idx)
    nsentences = len(train_lex_orig)
    numpy.random.seed(_args.seed)
    random.seed(_args.seed)
    _args.y_dim = nclasses
    _args.vocsize = len(feature2idx)  # ufnum #vocsize
    _args.in_dim = _args.vocsize  # + 2
    if _args.circuit == 'plainOrderOneCRF':
        _args.emission_trans_out_dim = nclasses
    _args.nsentences = nsentences
    # eval all training and topology related parameters
    for a in TOPO_PARAM + TRAIN_PARAM:
        try:
            _args.__dict__[a] = eval(_args.__dict__[a])
        except:
            pass
    # This way we can inject code from command line.
    if _args.use_emb == 'true':
        M_emb, idx_map = read_matrix_from_file(_args.emb_file, word2idx)
        emb_var = theano.shared(M_emb, name='emb_matrix')
        _args.emb_matrix = emb_var
        _args.emb_dim = len(M_emb[0])
        print 'embeding size:', _args.emb_dim
        if _args.fine_tuning == 'true':
            print 'fine tuning!!!!!'
            _args.emb_matrix.is_regularizable = True
    train_feat, train_lex, train_y = conv_data(
        train_feat, train_lex_orig, train_y, _args.win, _args.vocsize)
    valid_feat, valid_lex, valid_y = conv_data(
        valid_feat, valid_lex_orig, valid_y, _args.win, _args.vocsize)
    test_feat, test_lex, test_y = conv_data(
        test_feat, test_lex_orig, test_y, _args.win, _args.vocsize)
    best_f1 = -numpy.inf
    param = dict(clr=_args.lr, ce=0, be=0)  # Create Circuit
    (f_cost, f_update, f_classify, f_debug,
     cargs) = create_circuit(_args, StackConfig)
    #params_to_save = {k:v for k,v in cargs.items() if (hasattr(v, 'is_regularizable') and v.is_regularizable and k.startswith('tparam'))}
    # print _args
    idx2label = dict((k, v) for v, k in _args.label2idx.iteritems())
    idx2word = dict((k, v) for v, k in _args.word2idx.iteritems())
    groundtruth_valid = convert_id_to_word(valid_y, idx2label)
    groundtruth_test = None
    if _args.eval_test:
        groundtruth_test = convert_id_to_word(test_y, idx2label)
    epoch_id = -1
    while epoch_id + 1 < _args.nepochs:
        epoch_id += 1
        train(train_feat, train_lex, train_y, _args, f_cost,
              f_update, f_debug, epoch_id, param['clr'])
        # Train and Evaluate
        if epoch_id % _args.neval_epochs == 0:
            groundtruth_train = convert_id_to_word(train_y, idx2label)
            # print 'evaluate train!!!'
            res_train, pred_train = predict(
                train_feat, train_lex, idx2label, idx2word, _args, f_classify, groundtruth_train)
            # print 'evaluate valid!!!'
            res_valid, pred_valid = predict(
                valid_feat, valid_lex, idx2label, idx2word, _args, f_classify, groundtruth_valid)
            res_test, pred_test = predict(
                test_feat, test_lex, idx2label, idx2word, _args, f_classify, groundtruth_test)
            print('TEST: epoch', epoch_id,
                  'train F1', res_train['f1'],
                  'valid F1', res_valid['f1'],
                  #'test F1'   , res_test['f1']
                  )
            if _args.eval_test:
                print 'test F1', res_test['f1']
            # If this update created a 'new best' model then save it.
            if res_valid['f1'] > best_f1:
                best_f1 = res_valid['f1']
                param['be'] = epoch_id
                param['last_decay'] = epoch_id
                # res_train['f1'], , res_test['f1']
                param['vf1'] = (res_valid['f1'])
                # res_train['p'], , res_test['p']
                param['vp'] = (res_valid['p'])
                # res_train['r'], , res_test['r']
                param['vr'] = (res_valid['r'])
                if _args.eval_test:
                    param['tf1'] = (res_test['f1'])
                    param['tp'] = (res_test['p'])
                    param['tr'] = (res_test['r'])
                print "saving parameters!"
                cargs['f_classify'] = f_classify
                save_parameters(_args.save_model_param, cargs)
                '''
		print "loading parameters!"
		load_params(_args.save_model_param, cargs)
	        res_test, pred_test = predict(test_feat, test_lex, idx2label, idx2word, _args, f_classify, groundtruth_test)
	        print 'test F1:', res_test['f1']
		'''
            else:
                pass
        # decay learning rate if no improvement in 10 epochs
        # and (epoch_id - param['be']) % _args.decay_epochs == 0:
        if _args.decay and (epoch_id - param['last_decay']) >= _args.decay_epochs:
            print 'learning rate decay at epoch', epoch_id
            param['last_decay'] = epoch_id
            param['clr'] *= 0.5
        # If learning rate goes down to minimum then break.
        if param['clr'] < _args.minimum_lr:
            print "\nLearning rate became too small, breaking out of training"
            break

    print('BEST RESULT: epoch', param['be'],
          'valid F1', param['vf1'], param['vp'], param['vr'],
          # 'best test F1', param['tf1'], param['tp'], param['tr']
          )
    if _args.eval_test:
        print 'best test F1', param['tf1'], param['tp'], param['tr']

