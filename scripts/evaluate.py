#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-11 19:46:08

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import sys
import codecs as cs
from keras_src.evaluate_util import eval_ner


def read_corpus(corpus):
    with cs.open(corpus, 'r', 'utf-8') as src:
        labels = []
        src = src.read().strip().split('\n\n')
        for sent in src:
            label = []
            for pair in sent.strip().split('\n'):
                label.append(pair.split()[-1])
            labels.append(label)
        return labels


def evaluate(predictions, groundtruth=None):
    if groundtruth is None:
        return None, predictions
    results = eval_ner(predictions, groundtruth)
    return results, predictions


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Arguments ERROR!"
        print "Usage: python evaluate gold pred"
        sys.exit(-1)
    gold = read_corpus(sys.argv[1])
    pred = read_corpus(sys.argv[2])
    result, _ = evaluate(pred, gold)
    print "Test F1: %f, P: %f, R: %f" % (result['f1'], result['p'], result['r'])