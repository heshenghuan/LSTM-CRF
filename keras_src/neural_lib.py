#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-11 19:54:01

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import numpy as np
import codecs as cs
import keras
import matplotlib.pyplot as plt
from keras import backend as K


def read_matrix_from_file(fn, dic):
    '''
    Assume that the file contains words in first column,
    and embeddings in the rest and that dic maps words to indices.
    '''
    _data = open(fn).read().strip().split('\n')
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
        return (M * self.multiplier).astype(K.floatx())

    def __repr__(self):
        mults = ', multiplier=%s' % ((('%.3f' % self.multiplier)
                                      if type(self.multiplier) is float
                                      else str(self.multiplier)))
        mats = ((', matrix="%s"' % self.matrix_filename)
                if self.matrix_filename is not None
                else '')
        return "ArrayInit(ArrayInit.%s%s%s)" % (self.option, mults, mats)


class LossHistory(keras.callbacks.Callback):
    """写一个LossHistory类，保存loss和acc"""

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        # Because this script does not running on a PC but an Ubuntu server
        # So do not show the picture but save it.
        # plt.savefig('%s_loss-acc.png' % loss_type)

