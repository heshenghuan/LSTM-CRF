#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import sys
import time
import tensorflow as tf
from model import batch_index, neural_tagger


class Hybrid_LSTM_tagger(neural_tagger):
    """
    A LSTM+CRF tagger which used hybrid feature. Hybrid feature is a
    combination of traditional context feature and window-repr embeddings.
    """

    def __init__(self, nb_words, emb_dim, emb_matrix, feat_size, hidden_dim,
                 nb_classes, time_steps, fine_tuning=False, drop_rate=1.0,
                 batch_size=None, templates=1, window=1, l2_reg=0.):
        self.nb_words = nb_words
        self.emb_dim = emb_dim
        self.feat_size = feat_size
        self.hidden_dim = hidden_dim
        self.nb_classes = nb_classes
        self.fine_tuning = fine_tuning
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.templates = templates
        self.window = window
        self.l2_reg = l2_reg
        self.transition = None

        if self.fine_tuning:
            self.emb_matrix = tf.Variable(
                emb_matrix, dtype=tf.float32, name="embeddings")
        else:
            self.emb_matrix = tf.constant(
                emb_matrix, dtype=tf.float32, name="embeddings")

        with tf.name_scope('inputs'):
            self.F = tf.placeholder(
                tf.int32, shape=[None, self.time_steps, self.templates],
                name='F_placeholder')
            self.X = tf.placeholder(
                tf.int32, shape=[None, self.time_steps, self.window],
                name='X_placeholder')
            self.Y = tf.placeholder(
                tf.int32, shape=[None, self.time_steps],
                name='Y_placeholder')
            self.X_len = tf.placeholder(
                tf.int32, shape=[None, ], name='X_len_placeholder')
            self.keep_prob = tf.placeholder(tf.float32, name='output_dropout')
        self.build()
        return

    def __str__(self):
        return "Hybrid LSTM+CRF tagger"

    def build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[self.hidden_dim, self.nb_classes],
                initializer=tf.random_uniform_initializer(-0.2, 0.2),
                # initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='lstm_weights'
            )
            self.T = tf.get_variable(
                shape=[self.feat_size, self.nb_classes],
                initializer=tf.random_uniform_initializer(-0.2, 0.2),
                # initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='feat_weights'
            )
            self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
            # self.b = tf.get_variable(
            #     shape=[self.nb_classes],
            #     initializer=tf.truncated_normal_initializer(stddev=0.01),
            #     # initializer=tf.random_uniform_initializer(-0.2, 0.2),
            #     name="bias"
            # )
        return

    def inference(self, X, F, X_len, reuse=None):
        with tf.variable_scope('feat_score'):
            # sum of traditional feature values
            features = tf.nn.embedding_lookup(self.T, F)
            feat_sum = tf.reduce_sum(features, axis=2)
            feat_sum = tf.reshape(feat_sum, [-1, self.nb_classes])

        # get RNN outputs
        word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
        word_vectors = tf.nn.dropout(word_vectors, keep_prob=self.keep_prob)
        word_vectors = tf.reshape(
            word_vectors, [-1, self.time_steps, self.window * self.emb_dim])

        with tf.variable_scope('label_inference', reuse=reuse):
            outputs, _ = tf.nn.dynamic_rnn(
                self.lstm_fw,
                word_vectors,
                dtype=tf.float32,
                sequence_length=X_len
            )
            outputs = tf.reshape(outputs, [-1, self.hidden_dim])
            # outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        with tf.name_scope('softmax'):
            scores = feat_sum + tf.matmul(outputs, self.W) + self.b
            scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores

    def get_batch_data(self, x, f, y, l, batch_size, keep_prob=1.0, shuffle=True):
        for index in batch_index(len(y), batch_size, 1, shuffle):
            feed_dict = {
                self.X: x[index],
                self.Y: y[index],
                self.F: f[index],
                self.X_len: l[index],
                self.keep_prob: keep_prob,
            }
            yield feed_dict, len(index)

    def loss(self, pred):
        with tf.name_scope('loss'):
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
                pred, self.Y, self.X_len)
            cost = tf.reduce_mean(-log_likelihood)
            reg = tf.nn.l2_loss(self.T) + \
                tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
            if self.fine_tuning:
                reg += tf.nn.l2_loss(self.emb_matrix)
            cost += reg * self.l2_reg
            return cost

    def run(
        self,
        train_x, train_f, train_y, train_lens,
        valid_x, valid_f, valid_y, valid_lens,
        test_x, test_f, test_y, test_lens,
        FLAGS=None
    ):
        if FLAGS is None:
            print "FLAGS ERROR"
            sys.exit(0)

        self.lr = FLAGS.lr
        self.training_iter = FLAGS.train_steps
        self.train_file_path = FLAGS.train_data
        self.test_file_path = FLAGS.valid_data
        self.display_step = FLAGS.display_step

        # predication & cost-calculation
        pred = self.inference(self.X, self.F, self.X_len)
        cost = self.loss(pred)

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(cost, global_step=global_step)

        with tf.name_scope('saveModel'):
            localtime = time.strftime("%X %Y-%m-%d", time.localtime())
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = FLAGS.model_dir + localtime + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        with tf.name_scope('summary'):
            if FLAGS.log:
                localtime = time.strftime("%Y%m%d-%X", time.localtime())
                Summary_dir = FLAGS.log_dir + localtime

                info = 'batch{}, lr{}, l2_reg{}'.format(
                    self.batch_size, self.lr, self.l2_reg)
                info += ';' + self.train_file_path + ';' + \
                    self.test_file_path + ';' + 'Method:%s' % (self.__str__())
                train_acc = tf.placeholder(tf.float32)
                train_loss = tf.placeholder(tf.float32)
                summary_acc = tf.summary.scalar('ACC ' + info, train_acc)
                summary_loss = tf.summary.scalar('LOSS ' + info, train_loss)
                summary_op = tf.summary.merge([summary_loss, summary_acc])

                valid_acc = tf.placeholder(tf.float32)
                valid_loss = tf.placeholder(tf.float32)
                summary_valid_acc = tf.summary.scalar('ACC ' + info, valid_acc)
                summary_valid_loss = tf.summary.scalar(
                    'LOSS ' + info, valid_loss)
                summary_valid = tf.summary.merge(
                    [summary_valid_loss, summary_valid_acc])

                train_summary_writer = tf.summary.FileWriter(
                    Summary_dir + '/train')
                valid_summary_writer = tf.summary.FileWriter(
                    Summary_dir + '/valid')

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0

            if self.training_iter == 0:
                saver.restore(sess, FLAGS.restore_model)

            for epoch in xrange(self.training_iter):

                for train, num in self.get_batch_data(train_x, train_f, train_y, train_lens, self.batch_size, (1 - self.drop_rate)):
                    _, step, trans_matrix, loss, predication = sess.run(
                        [optimizer, global_step, self.transition, cost, pred],
                        feed_dict=train)
                    tags_seqs, _ = self.viterbi_decode(
                        num, predication, train[self.X_len], trans_matrix)
                    f = self.evaluate(
                        num, tags_seqs, train[self.Y], train[self.X_len])
                    if FLAGS.log:
                        summary = sess.run(summary_op, feed_dict={
                            train_loss: loss, train_acc: f})
                        train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, f)
                save_path = saver.save(sess, save_dir, global_step=step)
                print "[+] Model saved in file: %s" % save_path

                if epoch % self.display_step == 0:
                    rd, loss, acc = 0, 0., 0.
                    for valid, num in self.get_batch_data(valid_x, valid_f, valid_y, valid_lens, self.batch_size):
                        trans_matrix, _loss, predication = sess.run(
                            [self.transition, cost, pred], feed_dict=valid)
                        loss += _loss
                        tags_seqs, _ = self.viterbi_decode(
                            num, predication, valid[self.X_len], trans_matrix)
                        f = self.evaluate(
                            num, tags_seqs, valid[self.Y], valid[self.X_len])
                        acc += f
                        rd += 1
                    loss /= rd
                    acc /= rd
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                    if FLAGS.log:
                        summary = sess.run(summary_valid, feed_dict={
                            valid_loss: loss, valid_acc: acc})
                        valid_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: valid loss(avg)={:.6f}, acc(avg)={:.6f}'.format(step, loss, acc)
                    print 'round {}: max_acc={} BestIter={}\n'.format(epoch, max_acc, bestIter)
            print 'Optimization Finished!'
            # test process
            pred_test_y = []
            acc, loss, rd = 0., 0., 0
            for test, num in self.get_batch_data(test_x, test_f, test_y, test_lens, self.batch_size, shuffle=False):
                trans_matrix, _loss, predication = sess.run(
                    [self.transition, cost, pred], feed_dict=test)
                loss += _loss
                rd += 1
                tags_seqs, tags_scores = self.viterbi_decode(
                    num, predication, test[self.X_len], trans_matrix)
                f = self.evaluate(
                    num, tags_seqs, test[self.Y], test[self.X_len])
                acc += f
                pred_test_y.extend(tags_seqs)
            acc /= rd
            loss /= rd
            return pred_test_y, loss, acc
