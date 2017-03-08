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
import numpy as np
import codecs as cs
import tensorflow as tf
from keras_src.lstm_ner import lstm_ner
from keras_src.evaluate_util import eval_ner
from keras_src.train_util import read_matrix_from_file
from keras_src.constant import MAX_LEN
from keras_src.constant import MODEL_DIR, DATA_DIR, EMBEDDING_DIR, OUTPUT_DIR
from keras_src.pretreatment import pretreatment, unfold_corpus, conv_corpus


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_data', DATA_DIR + r'weiboNER.conll.train', 'Training data file')
tf.app.flags.DEFINE_string(
    'test_data', DATA_DIR + r'weiboNER.conll.test', 'Test data file')
tf.app.flags.DEFINE_string(
    'valid_data', DATA_DIR + r'weiboNER.conll.dev', 'Validation data file')
tf.app.flags.DEFINE_string('log_dir', OUTPUT_DIR, 'The log dir')
tf.app.flags.DEFINE_string('model_dir', MODEL_DIR, 'Models dir')
tf.app.flags.DEFINE_string("emb_dir", EMBEDDING_DIR, "Embeddings dir")
tf.app.flags.DEFINE_string("emb_type", "char", "Embeddings type: char/charpos")
tf.app.flags.DEFINE_integer("emb_dim", 100, "embedding size")
tf.app.flags.DEFINE_string("output_dir", OUTPUT_DIR, "Output dir")
tf.app.flags.DEFINE_string(
    "ner_feature_thresh", 0, "The minimum count OOV threshold for NER")
# tf.app.flags.DEFINE_boolean('only_test', False, 'Only do the test')
tf.app.flags.DEFINE_float("lr", 0.002, "learning rate")
tf.app.flags.DEFINE_float("keep_prob", 1., "dropout rate of hidden layer")
tf.app.flags.DEFINE_boolean(
    'fine_tuning', True, 'Whether fine-tuning the embeddings')
tf.app.flags.DEFINE_boolean(
    'eval_test', True, 'Whether evaluate the test data.')
tf.app.flags.DEFINE_integer("max_len", MAX_LEN,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("nb_classes", 15, "BMES")
tf.app.flags.DEFINE_integer("hidden_dim", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 100, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 50000, "trainning steps")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization weight")


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


def test_evaluate(sess, unary_score, test_sequence_length, transMatrix, inp,
                  tX, tY):
    totalEqual = 0
    batchSize = FLAGS.batch_size
    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] - 1) / batchSize) + 1
    correct_labels = 0
    total_labels = 0
    pred_labels = []
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = tY[i * batchSize:endOff]
        feed_dict = {inp: tX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)
        for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_sequence_length_val):
            # print("seg len:%d" % (sequence_length_))
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, transMatrix)
            # Evaluate word-level accuracy.
            pred_labels.append(viterbi_sequence)
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += sequence_length_
    accuracy = 100.0 * correct_labels / float(total_labels)
    print("Accuracy: %.2f%%" % accuracy)
    return pred_labels


def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.lr).minimize(total_loss)


def main(_):
    np.random.seed(1337)
    random.seed(1337)
    '''
    if FLAGS.only_test:
        cargs = {}
        print "#" * 67
        print "# Loading parameters!"
        print "#" * 67
        # loading model parameters form file.
        load_params(FLAGS.save_model_param, cargs)

        # Read raw corpus
        test_corpus, test_lens = read_corpus(
            FLAGS.test_data, cargs['emb_type'],
            anno=None, has_label=FLAGS.eval_test)
        # Unfold corpus into sentences part and labels part
        test_sentcs, test_labels = unfold_corpus(test_corpus)
        # Convert string word and label to corresponding index
        test_X, test_Y = conv_corpus(test_sentcs, test_labels,
                                     cargs['words2idx'], cargs['label2idx'],
                                     max_len=cargs['max_len'])
        idx2label = dict((k, v) for v, k in cargs['label2idx'].iteritems())
        # idx2word = dict((k, v) for v, k in cargs['words2idx'].iteritems())
        groundtruth_test = None
        if FLAGS.eval_test:
            groundtruth_test = test_labels
        model = lstm_ner()
        model.load()
        print "#" * 67
        print "# Model's summary:"
        print "#" * 67
        model.summary()
        model.set_optimizer(optimizer=cargs['optimizer'], lr=cargs['lr'])
        model.compile()

        pred_test = sequence_labeling(test_X, test_lens, cargs['trans'],
                                      cargs['inits'], model)
        pred_test_label = convert_id_to_word(pred_test, idx2label)
        res_test, pred_test_label = evaluate(pred_test_label, groundtruth_test)
        original_text = [[item[1] for item in sent] for sent in test_corpus]
        write_prediction(FLAGS.output_dir, original_text, pred_test_label)
        exit(0)'''

    print "#" * 67
    print "# Loading data from:"
    print "#" * 67
    print "Train:", FLAGS.train_data
    print "Valid:", FLAGS.valid_data
    print "Test: ", FLAGS.test_data
    # pretreatment process: read, split and create vocabularies
    train_set, valid_set, test_set, dicts, max_len = pretreatment(
        FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data,
        threshold=FLAGS.ner_feature_thresh, emb_type=FLAGS.emb_type,
        test_label=FLAGS.eval_test)

    # Reset the maximum sentence's length
    max_len = max(MAX_LEN, max_len)
    FLAGS.max_len = MAX_LEN

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
    FLAGS.label2idx = label2idx
    FLAGS.words2idx = words2idx
    FLAGS.feats2idx = feats2idx

    print "Lexical word size:     %d" % len(words2idx)
    print "Label size:            %d" % len(label2idx)
    print "-------------------------------------------------------------------"
    print "Training data size:    %d" % len(train_corpus)
    print "Validation data size:  %d" % len(valid_corpus)
    print "Test data size:        %d" % len(test_corpus)
    print "Maximum sentence len:  %d" % max_len

    # generate the transition and initial probability matrices
    # inits, trans = generate_prb(FLAGS.train_data, label2idx)
    # FLAGS.inits = inits
    # FLAGS.trans = trans

    # neural network's output_dim
    nb_classes = len(label2idx) + 1
    FLAGS.nb_classes = max(nb_classes, FLAGS.nb_classes)

    # Embedding layer's input_dim
    nb_words = len(words2idx)
    # FLAGS.nb_words = nb_words
    # FLAGS.in_dim = FLAGS.nb_words + 1

    # load embeddings from file
    print "#" * 67
    print "# Reading embeddings from file."
    print "#" * 67
    emb_file = EMBEDDING_DIR + 'weibo_%s_vectors' % (FLAGS.emb_type)
    emb_mat, idx_map = read_matrix_from_file(emb_file, words2idx)
    FLAGS.emb_dim = max(emb_mat.shape[1], FLAGS.emb_dim)
    print "embeddings' size:", emb_mat.shape
    if FLAGS.fine_tuning:
        print "The embeddings will be fine-tuned!"

    idx2label = dict((k, v) for v, k in FLAGS.label2idx.iteritems())
    # idx2words = dict((k, v) for v, k in FLAGS.words2idx.iteritems())

    # convert corpus from string to it's own index seq with post padding 0
    train_X, train_Y = conv_corpus(train_sentcs, train_labels,
                                   words2idx, label2idx, max_len=max_len)
    valid_X, valid_Y = conv_corpus(valid_sentcs, valid_labels,
                                   words2idx, label2idx, max_len=max_len)
    test_X, test_Y = conv_corpus(test_sentcs, test_labels,
                                 words2idx, label2idx, max_len=max_len)

    graph = tf.Graph()
    with graph.as_default():
        model = lstm_ner(nb_words, FLAGS.emb_dim, emb_mat, FLAGS.hidden_dim,
                         FLAGS.nb_classes, FLAGS.keep_prob, FLAGS.batch_size,
                         FLAGS.max_len, FLAGS.l2_reg, FLAGS.fine_tuning)
        total_loss = model.loss(train_X, train_Y)
        train_op = train(total_loss)
        test_unary_score, test_sequence_length = model.test_unary_score()
        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            training_step = FLAGS.train_steps
            for step in xrange(training_step):
                if sv.should_stop():
                    break
                try:
                    _, trans_matrix = sess.run([train_op, model.transition])
                    if step % 100 == 0:
                        print "Iter %d loss: [%r]" % (step, sess.run(total_loss))
                    if step % 1000 == 0:
                        test_evaluate(
                            sess, test_unary_score, test_sequence_length,
                            trans_matrix, model.X, valid_X, valid_Y
                        )
                except KeyboardInterrupt, e:
                    sv.saver.save(sess, FLAGS.model_dir, global_step=step + 1)
                    raise e
            sv.saver.save(sess, FLAGS.model_dir + "finnal_model")
            pred_test = test_evaluate(
                sess, test_unary_score, test_sequence_length, trans_matrix,
                model.X, test_X, test_Y
            )
            pred_test = [pred_test[i][:test_lens[i]] for i in xrange(len(pred_test))]
            pred_test_label = convert_id_to_word(pred_test, idx2label)
            res_test, pred_test_label = evaluate(pred_test_label, test_labels)
            print "Test F1: %f, P: %f, R: %f" % (res_test['f1'], res_test['p'], res_test['r'])
            original_text = [[item[0] for item in sent]
                             for sent in test_corpus]
            write_prediction(FLAGS.output_dir + 'prediction.utf8',
                             original_text, pred_test_label)
            sess.close()
    '''
    print "#" * 67
    print "# Training process start."
    print "#" * 67
    rd = 0
    best_f1 = float('-inf')
    # cargs = dict_from_argparse(FLAGS)
    param = dict(clr=FLAGS.lr, ce=0, be=0)
    groundtruth_train = train_labels
    groundtruth_valid = valid_labels
    groundtruth_test = None
    if FLAGS.eval_test:
        groundtruth_test = test_labels
    while rd < FLAGS.round:
        print "Training round: %d" % (rd + 1)
        # training process
        model.set_optimizer(optimizer=FLAGS.optimizer, lr=param['clr'])
        model.compile()
        model.fit(train_X, train_Y, batch_size=FLAGS.batch_size,
                  nb_epoch=FLAGS.nb_epochs, verbose=FLAGS.verbose,
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
        if FLAGS.eval_test:
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
            if FLAGS.eval_test:
                param['tf1'] = (res_test['f1'])
                param['tp'] = (res_test['p'])
                param['tr'] = (res_test['r'])
            print "saving parameters!"
            save_parameters(FLAGS.save_model_param, cargs)
            model.save()
        else:
            pass
        if FLAGS.decay and (rd - param['last_decay']) >= FLAGS.decay_epochs:
            print 'learning rate decay at round', rd
            param['last_decay'] = rd
            param['clr'] *= 0.5
        # If learning rate goes down to minimum then break.
        if param['clr'] < FLAGS.minimum_lr:
            print "\nLearning rate became too small, breaking out of training"
            break
        rd += 1
    print "#" * 67
    print "# End of training process."
    print "#" * 67
    print 'BEST RESULT rd', param['be'], ': valid F1: %f, P: %f, R: %f' % (param['vf1'], param['vp'], param['vr'])
    if FLAGS.eval_test:
        print 'test F1: %f, P: %f, R: %f' % (param['tf1'], param['tp'], param['tr'])
    '''


if __name__ == "__main__":
    tf.app.run()
