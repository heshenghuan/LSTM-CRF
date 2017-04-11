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
import tensorflow as tf
from constant import MAX_LEN
from train_util import batch_index
from features import templates


def init_variable(shape, name=None):
    initial = tf.random_uniform(shape, -0.01, 0.01)
    return tf.Variable(initial, name=name)


class neural_tagger():

    def __init__(self, nb_words, emb_dim, emb_matrix, hidden_dim, nb_classes,
                 keep_prob=1.0, batch_size=None, time_steps=MAX_LEN,
                 l2_reg=0., fine_tuning=False):
        self.nb_words = nb_words
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nb_classes = nb_classes
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
                tf.int32, shape=[None, self.time_steps, len(templates)],
                name='X_placeholder')
            self.Y = tf.placeholder(
                tf.int32, shape=[None, self.time_steps],
                name='Y_placeholder')
            self.X_len = tf.placeholder(
                tf.int32, shape=[None, ], name='X_len_placeholder')
            self.keep_prob = tf.placeholder(tf.float32, name='output_dropout')
        self.build()
        return

    def build():
        pass

    def inference(self, X, X_len, reuse=None):
        pass

    def get_batch_data(self, x, y, l, batch_size, keep_prob, shuffle=True):
        for index in batch_index(len(y), batch_size, 1, shuffle):
            feed_dict = {
                self.X: x[index],
                self.Y: y[index],
                self.X_len: l[index],
                self.keep_prob: keep_prob,
            }
            yield feed_dict, len(index)

    def test_unary_score(self):
        return self.inference(self.X, reuse=True)

    def loss(self, pred):
        with tf.name_scope('loss'):
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
                pred, self.Y, self.X_len)
            cost = tf.reduce_mean(-log_likelihood)
            reg = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
            if self.fine_tuning:
                reg += tf.nn.l2_loss(self.emb_matrix)
            cost += reg * self.l2_reg
            return cost

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

        # with tf.name_scope('loss'):
        #     log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
        #         pred, self.Y, self.X_len)
        #     cost = tf.reduce_mean(-log_likelihood)
        #     reg = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
        #     if self.fine_tuning:
        #         reg += tf.nn.l2_loss(self.emb_matrix)
        #     cost += reg * self.l2_reg
        cost = self.loss(pred)

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learn_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('summary'):
            localtime = time.strftime("%X %Y-%m-%d", time.localtime())
            Summary_dir = FLAGS.log_dir + localtime

            info = 'batch-{}, lr-{}, kb-{}, l2_reg-{}'.format(
                self.batch_size, self.learn_rate, self.Keep_Prob, self.l2_reg)
            info = info + '\n' + self.train_file_path + '\n' + \
                self.test_file_path + '\n' + 'Method: Bi-LSTM NER'
            train_acc = tf.placeholder(tf.float32)
            train_loss = tf.placeholder(tf.float32)
            summary_acc = tf.scalar_summary('ACC ' + info, train_acc)
            summary_loss = tf.scalar_summary('LOSS ' + info, train_loss)
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

            if self.training_iter == 0:
                saver.restore(sess, FLAGS.restore_model)

            for epoch in xrange(self.training_iter):

                for train, num in self.get_batch_data(train_x, train_y, train_lens, self.batch_size, self.Keep_Prob):
                    _, step, trans_matrix, loss, prediction = sess.run(
                        [optimizer, global_step, self.transition, cost, pred], feed_dict=train)
                    correct = 0.
                    cnt = sum(train[self.X_len])
                    prediction = np.argmax(prediction, 2)
                    for i in xrange(num):
                        p_len = train[self.X_len][i]
                        for j in xrange(p_len):
                            if prediction[i][j] == train[self.Y][i][j]:
                                correct += 1.0
                    acc = correct / cnt
                    summary = sess.run(summary_op, feed_dict={
                                       train_loss: loss, train_acc: acc})
                    train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                saver.save(sess, save_dir, global_step=step)

                if i % self.display_step == 0:
                    loss, cnt, rd = 0., 0, 0
                    correct = 0.0
                    for valid, num in self.get_batch_data(valid_x, valid_y, valid_lens, self.batch_size, keep_prob=1.0):
                        _loss, _prediction = sess.run(
                            [cost, pred], feed_dict=valid)
                        loss += _loss
                        rd += 1
                        cnt += sum(valid[self.X_len])
                        prediction = np.argmax(_prediction, 2)
                        for i in xrange(num):
                            p_len = valid[self.X_len][i]
                            for j in xrange(p_len):
                                if prediction[i][j] == valid[self.Y][i][j]:
                                    correct += 1.0
                    loss = loss / rd
                    acc = correct / cnt
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step

                    summary = sess.run(summary_test, feed_dict={
                                       test_loss: loss, test_acc: acc})
                    test_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: valid loss={:.6f}, valid acc={:.6f}'.format(step, loss, acc)
                    print 'round {}: max_acc={} BestIter={}\n'.format(epoch, max_acc, bestIter)
            print 'Optimization Finished!'
            pred_test_y = []
            loss, cnt = 0., 0
            correct = 0.0
            for test, num in self.get_batch_data(test_x, test_y, test_lens, self.batch_size, keep_prob=1.0, shuffle=False):
                prediction, trans_matrix, loss, = sess.run(
                    [pred, self.transition, cost], feed_dict=test)
                cnt += sum(test[self.X_len])
                for i in xrange(num):
                    p_len = test[self.X_len][i]
                    seq_scores = prediction[i][:p_len]
                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(
                        seq_scores, trans_matrix)
                    pred_test_y.append(viterbi_seq)
                    for j in xrange(p_len):
                        if viterbi_seq[j] == test[self.Y][i][j]:
                            correct += 1.0
            acc = correct / cnt
            return pred_test_y, loss, acc


class LSTM_NER(neural_tagger):

    def build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[self.hidden_dim, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights',
                regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )
            self.lstm_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
        return

    def inference(self, X, X_len, reuse=None):
        word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
        word_vectors = tf.reshape(
            word_vectors, [-1, self.time_steps, len(templates) * self.emb_dim])

        with tf.variable_scope('label_inference', reuse=reuse):
            outputs, _ = tf.nn.dynamic_rnn(
                self.lstm_layer,
                word_vectors,
                dtype=tf.float32,
                sequence_length=X_len
            )
            outputs = tf.reshape(outputs, [-1, self.hidden_dim])
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        with tf.name_scope('softmax'):
            scores = tf.matmul(outputs, self.W) + self.b
            scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores


class Bi_LSTM_NER(neural_tagger):

    def build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[self.hidden_dim * 2, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights',
                regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )
            self.lstm_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            self.lstm_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
        return

    def inference(self, X, X_len, reuse=None):
        word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
        word_vectors = tf.reshape(
            word_vectors, [-1, self.time_steps, len(templates) * self.emb_dim])

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
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        with tf.name_scope('softmax'):
            scores = tf.matmul(outputs, self.W) + self.b
            scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores


class CNN_Bi_LSTM_NER(neural_tagger):

    def build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[self.hidden_dim * 2, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights',
                regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )
            self.lstm_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            self.lstm_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            # extract bigram feature, so filter's shape :[1, 2*emb_dim, 1,
            # hidden_dim]
            self.conv_weight = tf.get_variable(
                shape=[2, self.emb_dim, 1, self.emb_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='conv_weights'
            )

        with tf.name_scope('biases'):
            self.b = init_variable([self.nb_classes], name="bias")
            self.conv_bias = init_variable([self.hidden_dim], name="conv_bias")
        return

    def inference(self, X, X_len, reuse=None):
        word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
        # word_vectors = tf.reshape(
        # word_vectors, [-1, self.time_steps, len(templates) * self.emb_dim])

        with tf.variable_scope('convolution'):
            word_vectors = tf.reshape(
                word_vectors, [-1, len(templates), self.emb_dim, 1])
            conv = tf.nn.conv2d(word_vectors, self.conv_weight,
                                strides=[1, 1, 1, 1], padding='VALID')
            conv = conv + self.conv_bias
            conv = tf.reshape(
                conv, [-1, self.time_steps, (len(templates) - 1) * self.emb_dim])
        word_vectors = tf.reshape(
            word_vectors, [-1, self.time_steps, len(templates) * self.emb_dim])
        word_vectors = tf.concat(2, [word_vectors, conv])

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
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        with tf.name_scope('softmax'):
            scores = tf.matmul(outputs, self.W) + self.b
            scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores

    def loss(self, pred):
        with tf.name_scope('loss'):
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
                pred, self.Y, self.X_len)
            cost = tf.reduce_mean(-log_likelihood)
            reg = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.conv_weight)\
                + tf.nn.l2_loss(self.b) + tf.nn.l2_loss(self.conv_bias)
            if self.fine_tuning:
                reg += tf.nn.l2_loss(self.emb_matrix)
            cost += reg * self.l2_reg
            return cost
