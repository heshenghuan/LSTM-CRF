#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-20 16:28:53

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import sys
import random
import argparse
import functools
import numpy as np
import codecs as cs
from keras_src.lstm_ner import lstm_ner
from keras_src.evaluate_util import eval_ner
from keras_src.train_util import load_params, save_parameters, read_matrix_from_file, sequence_labeling, dict_from_argparse
from keras_src.constant import MAX_LEN, BASE_DIR, MODEL_DIR, DATA_DIR, EMBEDDING_DIR
from keras_src.pretreatment import pretreatment, unfold_corpus, generate_prb, conv_corpus, read_corpus


def add_arg(name, default=None, **kwarg):
    assert hasattr(add_arg, 'arg_parser'), (
        "You must register an arg_parser with add_arg before calling it"
    )
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


def convert_id_to_word(corpus, idx2label):
    return [[idx2label.get(word, 'O') for word in sentence]
            for sentence
            in corpus]


def evaluate(predictions, groundtruth=None):
    if groundtruth is None:
        return None, predictions
    # conlleval(predictions, groundtruth,
    results = eval_ner(predictions, groundtruth)
    #          folder + '/current.valid.txt', folder)
    # error_analysis(words, predictions, groundtruth, idx2word)
    return results, predictions


def write_prediction(filename, lex_test, pred_test):
    with cs.open(filename, 'w', encoding='utf-8') as outf:
        for sent_w, sent_l in zip(lex_test, pred_test):
            assert len(sent_w) == len(sent_l)
            for w, l in zip(sent_w, sent_l):
                outf.write(w + '\t' + l + '\n')
            outf.write('\n')


def main(_args):
    np.random.seed(_args.seed)
    random.seed(_args.seed)
    if _args.only_test:
        cargs = {}
        print "###################################################################"
        print "# Loading parameters!"
        print "###################################################################"
        # loading model parameters form file.
        load_params(_args.save_model_param, cargs)
        # test_feat, test_lex_orig, test_y = get_data(
        #     _args.test_data, cargs['feature2idx'], cargs['word2idx'],
        #     cargs['label2idx'], cargs['emb_type'],
        #     anno=None, has_label=_args.eval_test)
        # Read raw corpus
        test_corpus, test_lens = read_corpus(
            _args.test_data, cargs['emb_type'],
            anno=None, has_label=_args.eval_test)
        # Unfold corpus into sentences part and labels part
        test_sentcs, test_labels = unfold_corpus(test_corpus)
        # Convert string word and label to corresponding index
        test_X, test_Y = conv_corpus(test_sentcs, test_labels,
                                     cargs['words2idx'], cargs['label2idx'],
                                     max_len=cargs['max_len'])
        idx2label = dict((k, v) for v, k in cargs['label2idx'].iteritems())
        idx2word = dict((k, v) for v, k in cargs['words2idx'].iteritems())
        groundtruth_test = None
        if _args.eval_test:
            groundtruth_test = test_labels
        model = lstm_ner()
        model.load()
        print "###################################################################"
        print "# Model's summary:"
        print "###################################################################"
        model.summary()
        model.set_optimizer(optimizer=cargs['optimizer'], lr=cargs['lr'])
        model.compile()

        pred_test = sequence_labeling(test_X, test_lens, cargs['trans'],
                                      cargs['inits'], model)
        pred_test_label = convert_id_to_word(pred_test, idx2label)
        res_test, pred_test_label = evaluate(pred_test_label, groundtruth_test)
        original_text = [[item[1] for item in sent] for sent in test_corpus]
        write_prediction(_args.output_dir, original_text, pred_test_label)
        exit(0)

    print "###################################################################"
    print "# Loading data from:"
    print "###################################################################"
    print "Train:", _args.training_data
    print "Valid:", _args.valid_data
    print "Test: ", _args.test_data
    # pretreatment process: read, split and create vocabularies
    train_set, valid_set, test_set, dicts, max_len = pretreatment(
        _args.training_data, _args.valid_data, _args.test_data,
        threshold=_args.ner_feature_thresh, emb_type=_args.emb_type,
        test_label=_args.eval_test)

    # Reset the maximum sentence's length
    max_len = max(MAX_LEN, max_len)
    _args.max_len = MAX_LEN

    # unfold these corpus
    train_corpus, train_lens = train_set
    valid_corpus, valid_lens = valid_set
    test_corpus, test_lens = test_set
    train_sentcs, train_labels = unfold_corpus(train_corpus)
    valid_sentcs, valid_labels = unfold_corpus(valid_corpus)
    test_sentcs, test_labels = unfold_corpus(test_corpus)

    # vocabularies
    feats2idx = dicts['feats2idx']
    words2idx = dicts['words2idx']
    label2idx = dicts['label2idx']
    _args.label2idx = label2idx
    _args.words2idx = words2idx
    _args.feats2idx = feats2idx

    print "Lexical word size:     %d" % len(words2idx)
    print "Label size:            %d" % len(label2idx)
    print "-------------------------------------------------------------------"
    print "Training data size:    %d" % len(train_corpus)
    print "Validation data size:  %d" % len(valid_corpus)
    print "Test data size:        %d" % len(test_corpus)
    print "Maximum sentence len:  %d" % max_len

    # generate the transition and initial probability matrices
    inits, trans = generate_prb(_args.training_data, label2idx)
    _args.inits = inits
    _args.trans = trans

    # neural network's output_dim
    nb_classes = len(label2idx) + 1
    _args.nb_classes = nb_classes

    # Embedding layer's input_dim
    nb_words = len(words2idx)
    _args.nb_words = nb_words
    _args.in_dim = _args.nb_words + 1

    # load embeddings from file
    print "###################################################################"
    print "# Reading embeddings from file."
    print "###################################################################"
    emb_mat, idx_map = read_matrix_from_file(_args.emb_file, words2idx)
    _args.emb_matrix = emb_mat
    _args.emb_dim = emb_mat.shape[1]
    print "embeddings' size:", emb_mat.shape
    if _args.fine_tuning:
        print "The embeddings will be fine-tuned!"

    idx2label = dict((k, v) for v, k in _args.label2idx.iteritems())
    # idx2words = dict((k, v) for v, k in _args.words2idx.iteritems())

    # convert corpus from string to it's own index seq with post padding 0
    train_X, train_Y = conv_corpus(train_sentcs, train_labels,
                                   words2idx, label2idx, max_len=max_len)
    valid_X, valid_Y = conv_corpus(valid_sentcs, valid_labels,
                                   words2idx, label2idx, max_len=max_len)
    test_X, test_Y = conv_corpus(test_sentcs, test_labels,
                                 words2idx, label2idx, max_len=max_len)

    model = lstm_ner()
    model.initialization(nb_words=nb_words, emb_dim=_args.emb_dim,
                         emb_matrix=emb_mat, output_dim=nb_classes,
                         batch_size=_args.batch_size, time_steps=max_len,
                         fine_tuning=_args.fine_tuning)
    print "###################################################################"
    print "# Model's summary:"
    print "###################################################################"
    model.summary()

    print "###################################################################"
    print "# Training process start."
    print "###################################################################"
    rd = 0
    best_f1 = float('-inf')
    cargs = dict_from_argparse(_args)
    param = dict(clr=_args.lr, ce=0, be=0)
    groundtruth_train = train_labels
    groundtruth_valid = valid_labels
    groundtruth_test = None
    if _args.eval_test:
        groundtruth_test = test_labels
    while rd < _args.round:
        print "Training round: %d" % (rd + 1)
        # training process
        model.set_optimizer(optimizer=_args.optimizer, lr=param['clr'])
        model.compile()
        model.fit(train_X, train_Y, batch_size=_args.batch_size,
                  nb_epoch=_args.nb_epochs, verbose=_args.verbose,
                  sequences_length=train_lens)

        # evaluate validation data
        pred_train = sequence_labeling(train_X, train_lens, trans, inits, model)
        pred_train_label = convert_id_to_word(pred_train, idx2label)
        res_train, pred_train_label = evaluate(pred_train_label, groundtruth_train)

        pred_valid = sequence_labeling(valid_X, valid_lens, trans, inits, model)
        pred_valid_label = convert_id_to_word(pred_valid, idx2label)
        res_valid, pred_valid_label = evaluate(pred_valid_label, groundtruth_valid)

        pred_test = sequence_labeling(test_X, test_lens, trans, inits, model)
        pred_test_label = convert_id_to_word(pred_test, idx2label)
        res_test, pred_test_label = evaluate(pred_test_label, groundtruth_test)

        print "Round", (rd + 1), ": train F1: %f, valid F1: %f" % (res_train['f1'], res_valid['f1'])
        if _args.eval_test:
            print 'test F1: %f' % res_test['f1']
        if res_valid['f1'] > best_f1:
            best_f1 = res_valid['f1']
            param['be'] = rd
            param['last_decay'] = rd
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
            save_parameters(_args.save_model_param, cargs)
            model.save()
        else:
            pass
        if _args.decay and (rd - param['last_decay']) >= _args.decay_epochs:
            print 'learning rate decay at round', rd
            param['last_decay'] = rd
            param['clr'] *= 0.5
        # If learning rate goes down to minimum then break.
        if param['clr'] < _args.minimum_lr:
            print "\nLearning rate became too small, breaking out of training"
            break
        rd += 1
    print "###################################################################"
    print "# End of training process."
    print "###################################################################"
    print 'BEST RESULT rd', param['be'], ': valid F1: %f, P: %f, R: %f' % (param['vf1'], param['vp'], param['vr'])
    if _args.eval_test:
        print 'test F1: %f, P: %f, R: %f' % (param['tf1'], param['tp'], param['tr'])


if __name__ == "__main__":
    ##########################################################################
    # PARSE ARGUMENTS, BUILD CIRCUIT, TRAIN, TEST
    ##########################################################################
    _arg_parser = argparse.ArgumentParser(description='LSTM-NER')
    add_arg.arg_parser = _arg_parser
    TOPO_PARAM = []
    TRAIN_PARAM = []
    # File IO
    add_arg('--only_test', False, help='Only do the test')
    add_arg('--save_model_param', MODEL_DIR+r'parameters.pkl',
            help='The best model will be saved there')
    add_arg('--training_data', DATA_DIR+r'weiboNER.conll.train',
            help='training file name')
    add_arg('--valid_data', DATA_DIR+r'weiboNER.conll.valid',
            help='validation file name')
    add_arg('--test_data', DATA_DIR+r'weiboNER.conll.test',
            help='test file name')
    add_arg('--output_dir', BASE_DIR+r'export/output.utf8',
            help='the output dir that stores the prediction')
    add_arg('--eval_test', True,
            help='Whether evaluate the test data: test data may not have annotations.')
    # make sure the type of embeddings file is exactly what you argument 'emb_type' gives
    add_arg('--emb_type', 'char',
            help='The embedding type, choose from (char, charpos)')
    add_arg('--emb_file', EMBEDDING_DIR+r'weibo_char_vectors',
            help='The initial embedding file name')
    add_arg('--ner_feature_thresh', 0,
            help="The minimum count OOV threshold for NER")
    # Training
    add_arg_to_L(TRAIN_PARAM, '--lr', 0.05)
    # add_arg_to_L(TRAIN_PARAM, '--use_emb', 'true')
    add_arg_to_L(TRAIN_PARAM, '--fine_tuning', True)
    add_arg_to_L(TRAIN_PARAM, '--nb_epochs', 200)
    add_arg_to_L(TRAIN_PARAM, '--round', 10)
    add_arg_to_L(TRAIN_PARAM, '--batch_size', 100)
    add_arg_to_L(TRAIN_PARAM, '--neval_epochs', 5)
    add_arg_to_L(TRAIN_PARAM, '--optimizer', 'rmsprop')
    add_arg_to_L(TRAIN_PARAM, '--seed', 1337, help='set random seed')
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
    add_arg('--verbose', 1)
    args = _arg_parser.parse_args()
    main(args)
    # from sighan_ner import loaddata, get_data, eval_ner, error_analysis  # , conlleval
    # main(args)
