#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-20 16:47:02

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import sys
import time
import numpy as np
import codecs as cs
# import cPickle as pickle
import tensorflow as tf
from constant import MAX_LEN
from train_util import batch_index


class LSTM_NER():

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


class Bi_LSTM_NER():

    def __init__(self, nb_words, emb_dim, emb_matrix, hidden_dim, nb_classes,
                 keep_prob=1.0, batch_size=None, time_steps=MAX_LEN,
                 l2_reg=0., fine_tuning=False):
        self.nb_words = nb_words
        self.emb_dim = emb_dim
        self.nb_classes = nb_classes
        self.hidden_dim = hidden_dim
        self.Keep_Prob = keep_prob
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.l2_reg = l2_reg
        self.fine_tuning = fine_tuning

        if self.fine_tuning:
            self.emb_matrix = tf.Variable(
                emb_matrix, dtype=tf.float32, name="embeddings")
        else:
            self.emb_matrix = tf.constant(
                emb_matrix, dtype=tf.float32, name="embeddings")

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.int32, shape=[None, self.time_steps], name='X_placeholder')
            self.Y = tf.placeholder(
                tf.int32, shape=[None, self.time_steps],
                name='Y_placeholder')
            self.X_len = tf.placeholder(
                tf.int32, shape=[None, ], name='X_len_placeholder')
            self.keep_prob = tf.placeholder(tf.float32, name='output_dropout')

        # def init_variable(shape):
        #     initial = tf.random_uniform(shape, -0.01, 0.01)
        #     return tf.Variable(initial)

        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[hidden_dim * 2, nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights',
                regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )
            self.lstm_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            self.lstm_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([nb_classes], name="bias"))
        return

    # def length(self, data):
    #     # 计算data句子中，非零元素的个数，也即计算句子长度
    #     used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    #     length = tf.reduce_sum(used, reduction_indices=1)
    #     length = tf.cast(length, tf.int32)
    #     return length

    def inference(self, X, X_len, reuse=None):
        word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
        # length = self.length(word_vectors)
        # length = np.asarray(lens, dtype='int32')
        with tf.variable_scope('label_inference', reuse=reuse):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.lstm_fw,
                cell_bw=self.lstm_bw,
                inputs=word_vectors,
                dtype=tf.float32,
                sequence_length=X_len
            )
            outputs = tf.concat(2, [outputs[0], outputs[1]])
            outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])

        with tf.name_scope('softmax'):
            # outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            # outputs = tf.reshape(outputs, [-1, self.emb_dim])
            scores = tf.matmul(outputs, self.W) + self.b
            scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores

    # def loss(self, X, Y):
    #     pred, lens = self.inference(X)
    #     log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
    #         pred, Y, lens)
    #     loss = tf.reduce_mean(-log_likelihood)
    #     reg = tf.nn.l2_loss(self.emb_matrix)
    #     reg += tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
    #     loss += reg * self.l2_reg
    #     return loss

    def test_unary_score(self):
        return self.inference(self.X, reuse=True)

    def get_batch_data(self, x, y, l, batch_size, keep_prob):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.X: x[index],
                self.Y: y[index],
                self.X_len: l[index],
                self.keep_prob: keep_prob,
            }
            yield feed_dict, len(index)

    def run(
        self,
        train_x, train_y, train_lens,
        valid_x, valid_y, valid_lens,
        test_x, test_y, test_lens,
        FLAGS=None
    ):
        if FLAGS is None:
            print "FLAGS ERROR"
            sys.exit(0)

        self.learn_rate = FLAGS.lr
        self.training_iter = FLAGS.train_steps
        self.train_file_path = FLAGS.train_data
        self.test_file_path = FLAGS.valid_data
        self.display_step = FLAGS.display_step
        pred = self.inference(self.X, self.X_len)

        with tf.name_scope('loss'):
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
                pred, self.Y, self.X_len)
            cost = tf.reduce_mean(-log_likelihood)
            reg = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
            if self.fine_tuning:
                reg += tf.nn.l2_loss(self.emb_matrix)
            cost += reg * self.l2_reg

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learn_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(pred, 2), tf.cast(self.Y, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        with tf.name_scope('summary'):
            localtime = time.strftime("%X %Y-%m-%d", time.localtime())
            Summary_dir = FLAGS.log_dir + localtime

            info = 'batch-{}, lr-{}, kb-{}, l2_reg-{}'.format(
                self.batch_size, self.learn_rate, self.Keep_Prob, self.l2_reg)
            info = info + '\n' + self.train_file_path + '\n' + \
                self.test_file_path + '\n' + 'Method: Bi-LSTM NER'
            summary_acc = tf.scalar_summary('ACC ' + info, accuracy)
            summary_loss = tf.scalar_summary('LOSS ' + info, cost)
            summary_op = tf.merge_summary([summary_loss, summary_acc])

            test_acc = tf.placeholder(tf.float32)
            test_loss = tf.placeholder(tf.float32)
            summary_test_acc = tf.scalar_summary('ACC ' + info, test_acc)
            summary_test_loss = tf.scalar_summary('LOSS ' + info, test_loss)
            summary_test = tf.merge_summary(
                [summary_test_loss, summary_test_acc])

            train_summary_writer = tf.train.SummaryWriter(
                Summary_dir + '/train')
            test_summary_writer = tf.train.SummaryWriter(Summary_dir + '/test')

        with tf.name_scope('saveModel'):
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = FLAGS.model_dir + localtime + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0

            for i in xrange(self.training_iter):

                for train, _ in self.get_batch_data(train_x, train_y, train_lens, self.batch_size, self.Keep_Prob):
                    _, step, summary, trans_matrix, loss, acc = sess.run(
                        [optimizer, global_step, summary_op, self.transition, cost, accuracy], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                saver.save(sess, save_dir, global_step=step)

                if i % self.display_step == 0:
                    acc, loss, cnt = 0., 0., 0
                    for valid, num in self.get_batch_data(valid_x, valid_y, valid_lens, len(valid_y), keep_prob=1.0):
                        _loss, _acc = sess.run(
                            [cost, correct_num], feed_dict=valid)
                        acc += _acc
                        loss += _loss * num
                        cnt += num * self.time_steps
                    loss = loss / cnt
                    acc = acc / cnt
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step

                    summary = sess.run(summary_test, feed_dict={
                                       test_loss: loss, test_acc: acc})
                    test_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: valid loss={:.6f}, valid acc={:.6f}'.format(step, loss, acc)
                    print 'round {}: max_acc={} BestIter={}\n'.format(i, max_acc, bestIter)
            print 'Optimization Finished!'
            pred_test_y = []
            acc, loss, cnt = 0., 0., 0
            for test, num in self.get_batch_data(test_x, test_y, test_lens, len(test_y), keep_prob=1.0):
                prediction, trans_matrix, _loss, _acc = sess.run([pred, self.transition, cost, correct_num], feed_dict=test)
                acc += _acc
                loss += _loss * num
                cnt += num * self.time_steps
                for i in xrange(len(prediction)):
                    seq_scores = prediction[i][:test_lens[i]]
                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(
                        seq_scores, trans_matrix)
                    pred_test_y.append(viterbi_seq)
            loss = loss / cnt
            acc = acc / cnt
            return pred_test_y, loss, acc
