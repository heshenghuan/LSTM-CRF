#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-20 16:47:02

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import numpy as np
import codecs as cs
# import cPickle as pickle
import tensorflow as tf
# from keras.models import Sequential, load_model
# from keras.layers import LSTM, Embedding, Dense
# from keras.optimizers import RMSprop, SGD
from constant import MAX_LEN


class lstm_ner():

    def __init__(self, nb_words, emb_dim, emb_matrix, hidden_dim, nb_classes,
                 keep_prob=1.0, batch_size=None, time_steps=MAX_LEN,
                 l2_reg=0., fine_tuning=False):
        """
        Returns an instance of lstm_ner.
        """
        def init_variable(shape):
            initial = tf.random_uniform(shape, -0.01, 0.01)
            return tf.Variable(initial)
        # self.model = None
        # self.embedding_layer = None
        # self.lstm_layer = None
        # self.output_layer = None
        # self.optimizer = None
        self.nb_words = nb_words
        self.emb_dim = emb_dim
        self.nb_classes = nb_classes
        self.emb_matrix = emb_matrix
        self.hidden_dim = hidden_dim
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.l2_reg = l2_reg
        self.fine_tuning = fine_tuning
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.int32,
                                    shape=[None, self.time_steps],
                                    name='X_placeholder')
            # self.seq_len = tf.placeholder(tf.int32,
            #                               shape=[batch_size, ],
            #                               name='seq_len_placeholder')
            # self.Y = tf.placeholder(tf.int32,
            #                         shape=[batch_size, time_steps, nb_classes],
            #                         name='Y_placeholder')
            # self.keep_prob = tf.placeholder(tf.float32, name='output_dropout')

        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[hidden_dim, nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights',
                regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )
            self.lstm_layer = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([nb_classes], name="bias"))
        return

    def length(self, data):
        # 计算data句子中，非零元素的个数，也即计算句子长度
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, X, reuse=None):
        word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
        length = self.length(word_vectors)
        # length = np.asarray(lens, dtype='int32')
        with tf.variable_scope('label_inference', reuse=reuse):
            outputs, _ = tf.nn.dynamic_rnn(
                self.lstm_layer,
                word_vectors,
                dtype=tf.float32,
                sequence_length=length
            )

        with tf.name_scope('softmax'):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            outputs = tf.reshape(outputs, [-1, self.emb_dim])
            scores = tf.matmul(outputs, self.W) + self.b
            scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores, length

    def loss(self, X, Y):
        pred, lens = self.inference(X)
        log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
            pred, Y, lens)
        loss = tf.reduce_mean(-log_likelihood)
        reg = tf.nn.l2_loss(self.emb_matrix)
        reg += tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
        loss += reg * self.l2_reg
        return loss

    def test_unary_score(self):
        pred, lens = self.inference(self.X, reuse=True)
        return pred, lens
