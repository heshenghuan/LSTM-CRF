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
from sklearn import metrics

__all__ = ['LSTM_NER', 'Bi_LSTM_NER', 'CNN_Bi_LSTM_NER']


def batch_index(length, batch_size, n_iter=100, shuffle=True):
    index = range(length)
    for j in xrange(n_iter):
        if shuffle:
            np.random.shuffle(index)
        for i in xrange(int(length / batch_size) + 1):
            yield index[i * batch_size: (i + 1) * batch_size]


def init_variable(shape, name=None):
    initial = tf.random_uniform(shape, -0.01, 0.01)
    return tf.Variable(initial, name=name)


class neural_tagger(object):
    """
    A tensorflow sequence labelling tagger.
    Use the 'build' method to customize the network structure of tagger.

    Example:
    1. LSTM + CRF: see class 'LSTM_NER'
    2. Bi-LSTM + CRF: see class 'Bi_LSTM_NER'
    3. CNN + Bi-LSTM + CRF: see class 'CNN_Bi_LSTM_NER'

    Use the 'inference' method to define how to calculate
    unary scores of given word sequence.

    Inherit this class and overwrite 'build' & 'inference', you can customize
    structure of your tagger.

    Then use 'run' method to training.
    """

    def __init__(self, nb_words, emb_dim, emb_matrix, hidden_dim, nb_classes,
                 drop_rate=1.0, batch_size=None, time_steps=0,
                 templates=1, l2_reg=0., fine_tuning=False):
        self.nb_words = nb_words
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nb_classes = nb_classes
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.templates = templates
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
                tf.int32, shape=[None, self.time_steps, self.templates],
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
        return "tagger"

    def build(self):
        pass

    def inference(self, X, X_len, reuse=None):
        pass

    def get_batch_data(self, x, y, l, batch_size, keep_prob=1.0, shuffle=True):
        for index in batch_index(len(y), batch_size, 1, shuffle):
            feed_dict = {
                self.X: x[index],
                self.Y: y[index],
                self.X_len: l[index],
                self.keep_prob: keep_prob,
            }
            yield feed_dict, len(index)

    def accuracy(self, num, pred, y, y_lens, trans_matrix):
        """
        Given predicted unary_scores, using viterbi_decode find the best tags
        sequence. Then count the correct labels and total labels.
        """
        correct_labels = 0
        total_labels = 0
        for i in xrange(num):
            p_len = y_lens[i]
            unary_scores = pred[i][:p_len]
            gold = y[i][:p_len]
            tags_seq, _ = tf.contrib.crf.viterbi_decode(
                unary_scores, trans_matrix)
            correct_labels += np.sum(np.equal(tags_seq, gold))
            total_labels += p_len
        return (correct_labels, total_labels)

    def viterbi_decode(self, num, pred, y_lens, trans_matrix):
        """
        Given predicted unary_scores, using viterbi_decode find the best tags
        sequence.
        """
        labels = []
        scores = []
        for i in xrange(num):
            p_len = y_lens[i]
            unary_scores = pred[i][:p_len]
            tags_seq, tags_score = tf.contrib.crf.viterbi_decode(
                unary_scores, trans_matrix)
            labels.append(tags_seq)
            scores.append(tags_score)
        return (labels, scores)

    def test_unary_score(self):
        """This method is deprecated."""
        return self.inference(self.X, self.X_len, reuse=True)

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

    def evaluate(self, num, labels, y, y_lens):
        golds = []
        preds = []
        for i in xrange(num):
            p_len = y_lens[i]
            golds.extend(y[i][:p_len])
            preds.extend(labels[i])
        # p = metrics.precision_score(golds, preds, average='macro')
        # r = metrics.recall_score(golds, preds, average='macro')
        # f = metrics.f1_score(golds, preds, average='macro')
        # return (p, r, f)
        return metrics.precision_score(golds, preds, average='micro')

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

        self.lr = FLAGS.lr
        self.training_iter = FLAGS.train_steps
        self.train_file_path = FLAGS.train_data
        self.test_file_path = FLAGS.valid_data
        self.display_step = FLAGS.display_step

        # unary_scores & loss
        pred = self.inference(self.X, self.X_len)
        cost = self.loss(pred)

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(cost, global_step=global_step)

        with tf.name_scope('summary'):
            if FLAGS.log:
                localtime = time.strftime("%Y%m%d-%X", time.localtime())
                Summary_dir = FLAGS.log_dir + localtime

                info = 'batch{}, lr{}, l2_reg{}'.format(
                    self.batch_size, self.lr, self.l2_reg)
                info += ';' + self.train_file_path + ';' + \
                    self.test_file_path + ';' + 'Method:%s' % self.__str__()
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

        with tf.name_scope('saveModel'):
            localtime = time.strftime("%X-%Y-%m-%d", time.localtime())
            saver = tf.train.Saver()
            save_dir = FLAGS.model_dir + localtime + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        with tf.Session() as sess:
            max_acc, bestIter = 0., 0

            if self.training_iter == 0:
                saver.restore(sess, FLAGS.restore_model)
                print "[+] Model restored from %s" % FLAGS.restore_model
            else:
                sess.run(tf.initialize_all_variables())

            for epoch in xrange(self.training_iter):

                for train, num in self.get_batch_data(train_x, train_y, train_lens, self.batch_size, (1 - self.drop_rate)):
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
                    for valid, num in self.get_batch_data(valid_x, valid_y, valid_lens, self.batch_size):
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
            pred_test_y = []
            acc, loss, rd = 0., 0., 0
            for test, num in self.get_batch_data(test_x, test_y, test_lens, self.batch_size, shuffle=False):
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


class LSTM_NER(neural_tagger):

    def __str__(self):
        return "LSTM-CRF NER"

    def build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[self.hidden_dim, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights'
            )
            self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
        return

    def inference(self, X, X_len, reuse=None):
        word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
        word_vectors = tf.nn.dropout(word_vectors, keep_prob=self.keep_prob)
        word_vectors = tf.reshape(
            word_vectors, [-1, self.time_steps, self.templates * self.emb_dim])

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
            scores = tf.matmul(outputs, self.W) + self.b
            # scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores


class Bi_LSTM_NER(neural_tagger):

    def __str__(self):
        return "BiLSTM-CRF NER"

    def build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[self.hidden_dim * 2, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights'
            )
            self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            self.lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
        return

    def inference(self, X, X_len, reuse=None):
        word_vectors = tf.nn.embedding_lookup(self.emb_matrix, X)
        word_vectors = tf.nn.dropout(word_vectors, keep_prob=self.keep_prob)
        word_vectors = tf.reshape(
            word_vectors, [-1, self.time_steps, self.templates * self.emb_dim])

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
            # outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        with tf.name_scope('softmax'):
            scores = tf.matmul(outputs, self.W) + self.b
            # scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores


class CNN_Bi_LSTM_NER(neural_tagger):

    def __str__(self):
        return "CNN-BiLSTM-CRF NER"

    def build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[self.hidden_dim * 2, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights'
            )
            self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            self.lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
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
        word_vectors = tf.nn.dropout(word_vectors, keep_prob=self.keep_prob)
        # word_vectors = tf.reshape(
        # word_vectors, [-1, self.time_steps, self.templates * self.emb_dim])

        with tf.variable_scope('convolution'):
            word_vectors = tf.reshape(
                word_vectors, [-1, self.templates, self.emb_dim, 1])
            conv = tf.nn.conv2d(word_vectors, self.conv_weight,
                                strides=[1, 1, 1, 1], padding='VALID')
            conv = conv + self.conv_bias
            conv = tf.reshape(
                conv, [-1, self.time_steps, (self.templates - 1) * self.emb_dim])
        word_vectors = tf.reshape(
            word_vectors, [-1, self.time_steps, self.templates * self.emb_dim])
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
            # outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        with tf.name_scope('softmax'):
            scores = tf.matmul(outputs, self.W) + self.b
            # scores = tf.nn.softmax(scores)
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
