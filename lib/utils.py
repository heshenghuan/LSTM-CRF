#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""


import os
import numpy as np
import codecs as cs
from parameters import FloatX

__all__ = [
    "read_emb_from_file", "conv_data"
]


class ArrayInit(object):
    """Returns an instance stored a NumPy array or None.
    # Usage
        Using the Initialize func to generate a matrix initialized by given
        option(see Initialize Options).

    # Initialize Options
        1. nomal, standard nomal distribution
        2. onesided_uniform
        3. twosided_uniform
        4. ortho, a random matrix after SVD
        5. zero
        6. unit
        7. ones
        8. fromfile, only use by given matrix and word2idx

    # Example
        ```python
        >>> M = ArrayInit(ArrayInit.nomal).initialize(10, 10)
        ```
        M is a 10*10 matrix where the elements x ~ N(0,1)
    """
    normal = 'normal'
    onesided_uniform = 'onesided_uniform'
    twosided_uniform = 'twosided_uniform'
    ortho = 'ortho'
    zero = 'zero'
    unit = 'unit'
    ones = 'ones'
    fromfile = 'fromfile'

    def __init__(self, option,
                 multiplier=1.0,
                 matrix=None,
                 word2idx=None):
        self.option = option
        self.multiplier = multiplier
        self.matrix_filename = None
        self.matrix = self._matrix_reader(matrix, word2idx)
        if self.matrix is not None:
            self.multiplier = 1
        return

    def _matrix_reader(self, matrix, word2idx):
        if type(matrix) is str:
            self.matrix_filename = matrix
            assert os.path.exists(matrix), "File %s not found" % matrix
            matrix = read_matrix_from_file(matrix, word2idx)
            return matrix
        else:
            return None

    def initialize(self, *xy, **kwargs):
        if self.option == ArrayInit.normal:
            # standard nomal distribution
            M = np.random.randn(*xy)
        elif self.option == ArrayInit.onesided_uniform:
            # a uniform distribution over [0, 1]
            M = np.random.rand(*xy)
        elif self.option == ArrayInit.twosided_uniform:
            # a uniform distribution over [-1, 1]
            M = np.random.uniform(-1.0, 1.0, xy)
        elif self.option == ArrayInit.ortho:
            # np.linalg.svd: singular value decomposition
            def f(dim):
                return np.linalg.svd(np.random.randn(dim, dim))[0]
            # f = lambda dim: np.linalg.svd(np.random.randn(dim, dim))[0]
            if int(xy[1] / xy[0]) < 1 and xy[1] % xy[0] != 0:
                raise ValueError(str(xy))
            M = np.concatenate(
                tuple(f(xy[0]) for _ in range(int(xy[1] / xy[0]))), axis=1)
            assert M.shape == xy
        elif self.option == ArrayInit.zero:
            M = np.zeros(xy)
        elif self.option in [ArrayInit.unit, ArrayInit.ones]:
            M = np.ones(xy)
        elif self.option == ArrayInit.fromfile:
            assert isinstance(self.matrix, np.ndarray)
            M = self.matrix
        else:
            raise NotImplementedError
        self.multiplier = (kwargs['multiplier']
                           if ('multiplier' in kwargs
                               and kwargs['multiplier'] is not None)
                           else self.multiplier)
        return (M * self.multiplier).astype(FloatX)

    def __repr__(self):
        mults = ', multiplier=%s' % ((('%.3f' % self.multiplier)
                                      if type(self.multiplier) is float
                                      else str(self.multiplier)))
        mats = ((', matrix="%s"' % self.matrix_filename)
                if self.matrix_filename is not None
                else '')
        return "ArrayInit(ArrayInit.%s%s%s)" % (self.option, mults, mats)


def read_matrix_from_file(fn, dic):
    '''
    Assume that the file contains words in first column,
    and embeddings in the rest and that dic maps words to indices.
    '''
    _data = cs.open(fn, 'r').read().strip().split('\n')
    _data = [e.strip().split() for e in _data]
    dim = len(_data[0]) - 1
    data = {}
    # NOTE: The norm of onesided_uniform rv is sqrt(n)/sqrt(3)
    # Since the expected value of X^2 = 1/3 where X ~ U[0, 1]
    # => sum(X_i^2) = dim/3
    # => norm       = sqrt(dim/3)
    # => norm/dim   = sqrt(1/3dim)
    multiplier = np.sqrt(1.0 / (3 * dim))
    for e in _data:
        r = np.array([float(_e) for _e in e[1:]])
        data[e[0]] = (r / np.linalg.norm(r)) * multiplier
    M = ArrayInit(ArrayInit.onesided_uniform, multiplier=1.0 /
                  dim).initialize(len(data), dim)
    for word, idx in dic.iteritems():
        if word in data:
            M[idx] = data[word]
    return M


def read_emb_from_file(fn, dic, ecd='utf-8'):
    """
    Reads embedding matrix from file.

    ## Parameters
        fn: embeddings file path
        dic: a vocabulary stored words you need and their indices
        ecd: file encoding

    ## Return
        M:
            The embeddings matrix of shape(len(dic), emb_dim),
            this means M does not include all embeddings stored in fn.
            Only include the embeddings of words appeared in dic.
        idx_map:
            A dict stored the original line num of a word index that in dic.
            Might be useless.
    """
    # multiplier = np.sqrt(1.0 / 3)
    not_in_dict = 0
    with cs.open(fn, encoding=ecd, errors='ignore') as inf:
        row, column = inf.readline().rstrip().split()
        dim = int(column)
        # print row, column
        idx_map = dict()
        line_count = 0
        M = ArrayInit(ArrayInit.normal).initialize(len(dic) + 1, dim)
        for line in inf:
            elems = line.rstrip().split(' ')
            if elems[0] in dic:
                idx = dic[elems[0]]
                vec_elem = elems[1:]
                r = np.array([float(_e) for _e in vec_elem])
                # np.linalg.norm: 求范数
                # M[idx] = (r / np.linalg.norm(r)) * multiplier
                M[idx] = r
                idx_map[idx] = line_count
            else:
                not_in_dict += 1
            line_count += 1
        print 'load embedding! %s words,' % (row),
        print '%d not in the dictionary.' % (not_in_dict),
        print ' Dictionary size: %d' % (len(dic))
        # print M.shape, len(idx_map)
        return M, idx_map


def conv_data(feature_arry, lex_arry, label_arry, win_size, feat_size):
    """
    Converts features, lexs, labels(all of them are lists of idxs) into
    int32 ndarray.
    """
    fv = []
    lexv = []
    labv = []
    for i, (f, x, y) in enumerate(zip(feature_arry, lex_arry, label_arry)):
        words = x  # _conv_x(x, M_emb, win_size)
        features = _conv_feat(f, feat_size)
        labels = _conv_y(y)
        fv.append(features)
        lexv.append(words)
        labv.append(labels)
    return fv, lexv, labv


def _conv_feat(x, feat_size):
    """
    Converts the features list into a int32 array.
    """
    lengths = [len(elem) for elem in x]
    max_len = max(lengths)
    features = np.ndarray(len(x), max_len).astype('int32')
    for i, feat in enumerate(x):
        fpadded = feat + [feat_size] * (max_len - len(feat))
        features[i] = fpadded
    return features


def _conv_y(y):
    """
    Converts the labels list to a int32 arrary.
    """
    labels = np.ndarray(len(y)).astype('int32')
    for i, label in enumerate(y):
        labels[i] = label
    return labels


def eval_ner(pred, gold):
    """
    Utilities for evaluation.
    2017-02-15 19:29:49 还未阅读这部分
    """
    print 'Evaluating...'
    eval_dict = {}    # value=[#match, #pred, #gold]
    for p_1sent, g_1sent in zip(pred, gold):
        in_correct_chunk = False
        last_pair = ['^', '$']
        for p, g in zip(p_1sent, g_1sent):
            tp = p.split('-')
            tg = g.split('-')
            if len(tp) == 2:
                if tp[1] not in eval_dict:
                    eval_dict[tp[1]] = [0] * 3
                if tp[0] == 'B' or tp[0] == 'S':
                    eval_dict[tp[1]][1] += 1
            if len(tg) == 2:
                if tg[1] not in eval_dict:
                    eval_dict[tg[1]] = [0] * 3
                if tg[0] == 'B' or tg[0] == 'S':
                    eval_dict[tg[1]][2] += 1

            if p != g or len(tp) == 1:
                if in_correct_chunk and tp[0] != 'I' and tg[0] != 'I' and tp[0] != 'E' and tg[0] != 'E':
                    assert last_pair[0] == last_pair[1]
                    eval_dict[last_pair[0]][0] += 1
                in_correct_chunk = False
                last_pair = ['^', '$']
            else:
                if tg[0] == 'B' or tg[0] == 'S':
                    if in_correct_chunk:
                        assert (last_pair[0] == last_pair[1])
                        eval_dict[last_pair[0]][0] += 1
                    last_pair = [tp[-1], tg[-1]]
                if tg[0] == 'B':
                    in_correct_chunk = True
                if tg[0] == 'S':
                    eval_dict[last_pair[0]][0] += 1
                    in_correct_chunk = False
        if in_correct_chunk:
            assert last_pair[0] == last_pair[1]
            eval_dict[last_pair[0]][0] += 1
    agg_measure = [0.0] * 3
    agg_counts = [0] * 3
    for k, v in eval_dict.items():
        agg_counts = [sum(x) for x in zip(agg_counts, v)]
        prec = float(v[0]) / v[1] if v[1] != 0 else 0.0
        recall = float(v[0]) / v[2] if v[2] != 0 else 0.0
        F1 = 2 * prec * recall / \
            (prec + recall) if prec != 0 and recall != 0 else 0.0
        agg_measure[0] += prec
        agg_measure[1] += recall
        agg_measure[2] += F1
        print k + ':', v[0], '\t', v[1], '\t', v[2], '\t', prec, '\t', recall, '\t', F1
    agg_measure = [v / len(eval_dict) for v in agg_measure]
    print 'Macro average:', '\t', agg_measure[0], '\t', agg_measure[1], '\t', agg_measure[2]
    prec = float(agg_counts[0]) / agg_counts[1] if agg_counts[1] != 0 else 0.0
    recall = float(agg_counts[0]) / \
        agg_counts[2] if agg_counts[2] != 0 else 0.0
    F1 = 2 * prec * recall / \
        (prec + recall) if prec != 0 and recall != 0 else 0.0
    print 'Micro average:', agg_counts[0], '\t', agg_counts[1], '\t', agg_counts[2], '\t', prec, '\t', recall, '\t', F1
    return {'p': prec, 'r': recall, 'f1': F1}
