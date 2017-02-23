#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-22 21:01:21

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""


import os
import sys
import time
import jieba
import random
import codecs as cs
import numpy as np
import cPickle as pickle
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from constant import OOV, local_templates, MAX_LEN
from features import escape, readiter, feature_extractor

random.seed(1337)


def create_dicts(train_fn, valid_fn, test_fn, threshold, mode, anno=None):
    """
    Returns three dicts, which are feature dictionary, word dictionary
    and label dictionary.

    ## Params
        train_fn: train data file path
        valid_fn: valid data file path
        test_fn: test data file path
        threshold: threshold value of feature frequency
        mode: the type of embeddings(char/charpos)
        anno: whether the data is annotated
    ## Return
        returns 3 dicts, and each is a map from term to index.
    """
    train = cs.open(train_fn, encoding='utf-8').read().strip().split('\n\n')
    if valid_fn is not None:
        valid = cs.open(
            valid_fn, encoding='utf-8').read().strip().split('\n\n')
    if test_fn is not None:
        test = cs.open(
            test_fn, encoding='utf-8').read().strip().split('\n\n')

    def get_label(stream, pos):
        return [[e.split()[pos]for e in line.strip().split('\n')]
                for line in stream]

    words = (get_label(train, 0)  # 训练集字符集合
             + ([] if valid_fn is None else get_label(valid, 0))  # 验证集字符集合
             + ([] if test_fn is None else get_label(test, 0)))  # 测试集字符集合
    labels = (get_label(train, -1)
              + ([] if valid_fn is None else get_label(valid, -1))
              + ([] if test_fn is None else get_label(test, -1)))
    corpus_feats = []
    corpus_words = []
    for lwds, llbs in zip(words, labels):
        X = convdata_helper(lwds, llbs, mode, anno)
        features = apply_feature_templates(X)
        feats = []
        for char in X:
            corpus_words.append(char[1])
        for i, ftv in enumerate(features):
            # escape函数将ft中的':'替换成'__COLON__'
            feat = [escape(ft) for ft in ftv['F']]
            feats.append(feat)
        assert len(lwds) == len(llbs)
        assert len(lwds) == len(feats)
        corpus_feats.append(feats)
    feature_to_freq = defaultdict(int)
    for feats in corpus_feats:  # feats一句话的特征列表
        for ftv in feats:  # ftv一句中某个字的特征集合
            for ft in ftv:  # ft这句话中某个字的按模板生成的某个特征
                feature_to_freq[ft] += 1
    features_to_id = {OOV: 1}
    cur_idx = 2
    for feats in corpus_feats:
        for ftv in feats:
            for ft in ftv:
                if (ft not in features_to_id):
                    if feature_to_freq[ft] > threshold:
                        # 只保留出现次数大于一定频次的特征
                        features_to_id[ft] = cur_idx
                        cur_idx += 1

    word_to_id = {}
    cur_idx = 1
    # print 'construct dict!!'
    for t in corpus_words:
        if t not in word_to_id:
            # print t, cur_idx
            word_to_id[t] = cur_idx
            cur_idx += 1

    label_to_id = {}
    cur_idx = 1
    for ilabels in labels:
        for t in ilabels:
            if t not in label_to_id:
                label_to_id[t] = cur_idx
                cur_idx += 1
    return features_to_id, word_to_id, label_to_id


def convdata_helper(chars, labels, repre, anno):
    """
    将chars和labels按照repre指定的形式处理：c为字符，w为词，l为标注

    1.repre == char: X = [(c, c, l) * n]
    2.repre == charpos: X = [(c, c+pos, l) * n]
    """
    X = []
    if repre == 'char' and anno is None:
        for c, l in zip(chars, labels):
            X.append((c, c, l))
    else:
        sent = ''.join(chars)
        token_tag_list = jieba.cut(sent)  # pseg.cut(sent)
        count = 0
        # for token_str, pos_tag in token_tag_list:
        for token_str in token_tag_list:
            for i, char in enumerate(token_str):
                if repre == 'charpos':
                    r = char + str(i)
                else:
                    raise ValueError(
                        'representation cannot take value %s!\n' % repre)
                if anno is None:
                    fields = (char, r, labels[count])
                else:
                    raise ValueError(
                        'annotation cannot take value %s!\n' % anno)
                count += 1
                X.append(fields)
    return X


def apply_feature_templates(sntc):
    """
    对句子应用特征模板抽取特征，会将句子的变量的类型改为dict
    sntc原始为(w,r,y)的列表，readiter会将其变成一个字典，有w,r,y和F三个域
    分别存储原来的char,repre,label和抽取的特征
    """
    if len(sntc[0]) == 3:
        fields_template = 'w r y'
    # elif len(sntc[0]) == 4:
    #     fields_template = 'w r s y'
    # elif len(sntc[0]) == 5:
    #     fields_template = 'w r s p y'
    else:
        raise ValueError('unknow representation!\n')
    if fields_template == 'w r y':
        features = feature_extractor(
            readiter(sntc, fields_template.split(' ')),
            templates=local_templates)
    else:
        features = feature_extractor(
            readiter(sntc, fields_template.split(' ')))
    return features


def generate_prb(train_fn, labels2idx):
    """
    Returns initial probability matrix and transition probability matrix.
    """
    train = cs.open(train_fn, encoding='utf-8').read().strip().split('\n\n')

    def get_label(stream, pos):
        return [[e.split()[pos]for e in line.strip().split('\n')]
                for line in stream]

    size = len(labels2idx)
    labels = get_label(train, -1)
    trans = np.zeros((size, size))
    inits = np.zeros((size))
    for line in labels:
        inits[labels2idx[line[0]]] += 1
        for i in range(1, len(line)):
            idx1 = labels2idx[line[i - 1]]
            idx2 = labels2idx[line[i]]
            trans[idx1][idx2] += 1
    sum_init = sum(inits)
    inits = np.log(inits / sum_init)
    for i in range(size):
        sum_i = sum(trans[i])
        trans[i] = np.log(trans[i] / sum_i)
    return inits, trans


def conv_sntc(X, Y, word2idx, label2idx):
    sntc = [word2idx.get(w) for w in X]
    label = [label2idx.get(l) for l in Y]
    return sntc, label


def conv_corpus(sntcs, labels, word2idx, label2idx):
    assert len(sntcs) == len(
        labels), "The length of input sentences and labels not equal."
    new_sntcs = []
    new_labels = []
    for sntc, label in zip(sntcs, labels):
        sntc, label = conv_sntc(sntc, label, word2idx, label2idx)
        new_sntcs.append(sntc)
        new_labels.append(label)
    new_sntcs = pad_sequences(new_sntcs, maxlen=MAX_LEN, padding='post')
    new_labels = to_categorical(np.asarray(new_labels))
    # new_labels = pad_sequences(new_labels, maxlen=MAX_LEN, padding='post')
    return new_sntcs, new_labels
