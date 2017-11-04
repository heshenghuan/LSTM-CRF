#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import random
import numpy as np
import codecs as cs
import tensorflow as tf
from hybrid_model import Hybrid_LSTM_tagger
from lib.src.parameters import MAX_LEN
from lib.src.features import HybridTemplate
from lib.src.utils import eval_ner, read_emb_from_file
from lib.src.pretreatment import pretreatment, unfold_corpus
from lib.src.pretreatment import conv_corpus, read_corpus
from env_settings import MODEL_DIR, DATA_DIR, EMB_DIR, OUTPUT_DIR, LOG_DIR


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_data', DATA_DIR + r'weiboNER.conll.train', 'Training data file')
tf.app.flags.DEFINE_string(
    'test_data', DATA_DIR + r'weiboNER.conll.test', 'Test data file')
tf.app.flags.DEFINE_string(
    'valid_data', DATA_DIR + r'weiboNER.conll.dev', 'Validation data file')
tf.app.flags.DEFINE_string('log_dir', LOG_DIR, 'The log dir')
tf.app.flags.DEFINE_string('model_dir', MODEL_DIR, 'Models dir')
tf.app.flags.DEFINE_string('restore_model', 'None',
                           'Path of the model to restored')
tf.app.flags.DEFINE_string(
    "emb_file", EMB_DIR + "/weibo_charpos_vectors", "Embeddings file")
tf.app.flags.DEFINE_integer("emb_dim", 100, "embedding size")
tf.app.flags.DEFINE_string("output_dir", OUTPUT_DIR, "Output dir")
tf.app.flags.DEFINE_boolean('only_test', False, 'Only do the test')
tf.app.flags.DEFINE_float("lr", 0.002, "learning rate")
tf.app.flags.DEFINE_float("dropout", 0., "Dropout rate of input layer")
tf.app.flags.DEFINE_boolean(
    'fine_tuning', True, 'Whether fine-tuning the embeddings')
tf.app.flags.DEFINE_boolean(
    'eval_test', True, 'Whether evaluate the test data.')
# tf.app.flags.DEFINE_boolean(
#     'test_anno', True, 'Whether the test data is labeled.')
tf.app.flags.DEFINE_integer("max_len", MAX_LEN,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("nb_classes", 15, "Tagset size")
tf.app.flags.DEFINE_integer("hidden_dim", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 200, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 50, "trainning steps")
tf.app.flags.DEFINE_integer("display_step", 1, "number of test display step")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization weight")
tf.app.flags.DEFINE_boolean(
    'log', True, 'Whether to record the TensorBoard log.')
tf.app.flags.DEFINE_string("template", r"template", "Feature templates")
tf.app.flags.DEFINE_integer("window", 1, "Window size of context")


def convert_id_to_word(corpus, idx2label, default='O'):
    return [[idx2label.get(word, default) for word in sentence]
            for sentence in corpus]


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


def save_dicts(path, feats2idx, words2idx, label2idx):
    with cs.open(path + 'FEATS', 'w', 'utf-8') as out:
        for k, v in feats2idx.iteritems():
            out.write("%s %d\n" % (k, v))

    with cs.open(path + 'WORDS', 'w', 'utf-8') as out:
        for k, v in words2idx.iteritems():
            out.write("%s %d\n" % (k, v))

    with cs.open(path + 'LABEL', 'w', 'utf-8') as out:
        for k, v in label2idx.iteritems():
            out.write("%s %d\n" % (k, v))


def load_dicts(path):
    feats2idx = {}
    words2idx = {}
    label2idx = {}
    with cs.open(path + 'FEATS', 'r', 'utf-8') as src:
        items = src.read().strip().split('\n')
        for item in items:
            k, v = item.strip().split()
            feats2idx[k] = int(v)

    with cs.open(path + 'WORDS', 'r', 'utf-8') as src:
        items = src.read().strip().split('\n')
        for item in items:
            k, v = item.strip().split()
            words2idx[k] = int(v)

    with cs.open(path + 'LABEL', 'r', 'utf-8') as src:
        items = src.read().strip().split('\n')
        for item in items:
            k, v = item.strip().split()
            label2idx[k] = int(v)

    return feats2idx, words2idx, label2idx


def main(_):
    np.random.seed(1337)
    random.seed(1337)

    if FLAGS.only_test or FLAGS.train_steps == 0:
        FLAGS.train_steps = 0
        test(FLAGS)
        return

    print "#" * 67
    print "# Loading data from:"
    print "#" * 67
    print "Train:", FLAGS.train_data
    print "Valid:", FLAGS.valid_data
    print "Test: ", FLAGS.test_data

    if FLAGS.window == 1:
        win = (0, 0)
    elif FLAGS.window == 3:
        win = (-1, 1)
    elif FLAGS.window == 5:
        win = (-2, 2)
    else:
        raise ValueError('Unsupported window size %d.' % FLAGS.window)

    # Choose fields templates & features templates
    template = HybridTemplate(FLAGS.template, win)
    # pretreatment process: read, split and create vocabularies
    train_set, valid_set, test_set, dicts, max_len = pretreatment(
        FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data,
        threshold=0, template=template)

    # Reset the maximum sentence's length
    # max_len = max(MAX_LEN, max_len)
    FLAGS.max_len = max_len

    # unfold these corpus
    train_corpus, train_lens = train_set
    valid_corpus, valid_lens = valid_set
    test_corpus, test_lens = test_set
    train_sentcs, train_featvs, train_labels = unfold_corpus(train_corpus)
    valid_sentcs, valid_featvs, valid_labels = unfold_corpus(valid_corpus)
    test_sentcs, test_featvs, test_labels = unfold_corpus(test_corpus)

    # vocabularies
    feats2idx = dicts['feats2idx']
    words2idx = dicts['words2idx']
    label2idx = dicts['label2idx']
    FLAGS.label2idx = label2idx
    FLAGS.words2idx = words2idx
    FLAGS.feats2idx = feats2idx
    FLAGS.feat_size = len(feats2idx)

    print "Lexical word size:     %d" % len(words2idx)
    print "Label size:            %d" % len(label2idx)
    print "Features size:         %d" % len(feats2idx)
    print "-------------------------------------------------------------------"
    print "Training data size:    %d" % len(train_corpus)
    print "Validation data size:  %d" % len(valid_corpus)
    print "Test data size:        %d" % len(test_corpus)
    print "Maximum sentence len:  %d" % FLAGS.max_len

    del train_corpus
    del valid_corpus
    # del test_corpus

    # neural network's output_dim
    nb_classes = len(label2idx)
    FLAGS.nb_classes = nb_classes + 1

    # Embedding layer's input_dim
    nb_words = len(words2idx)
    FLAGS.nb_words = nb_words
    FLAGS.in_dim = FLAGS.nb_words + 1

    # load embeddings from file
    print "#" * 67
    print "# Reading embeddings from file: %s" % (FLAGS.emb_file)
    emb_mat, idx_map = read_emb_from_file(FLAGS.emb_file, words2idx)
    FLAGS.emb_dim = max(emb_mat.shape[1], FLAGS.emb_dim)
    print "embeddings' size:", emb_mat.shape
    if FLAGS.fine_tuning:
        print "The embeddings will be fine-tuned!"

    idx2label = dict((k, v) for v, k in FLAGS.label2idx.iteritems())
    # idx2words = dict((k, v) for v, k in FLAGS.words2idx.iteritems())

    # convert corpus from string to it's own index seq with post padding 0
    print "Preparing training, validate and testing data."
    train_X, train_F, train_Y = conv_corpus(
        train_sentcs, train_featvs, train_labels,
        words2idx, feats2idx, label2idx, max_len=max_len)
    valid_X, valid_F, valid_Y = conv_corpus(
        valid_sentcs, valid_featvs, valid_labels,
        words2idx, feats2idx, label2idx, max_len=max_len)
    test_X, test_F, test_Y = conv_corpus(
        test_sentcs, test_featvs, test_labels,
        words2idx, feats2idx, label2idx, max_len=max_len)

    del train_sentcs, train_featvs, train_labels
    del valid_sentcs, valid_featvs, valid_labels
    # del test_sentcs, test_featvs, test_labels

    print "#" * 67
    print "Training arguments"
    print "#" * 67
    print "L2 regular:    %f" % FLAGS.l2_reg
    print "nb_classes:    %d" % FLAGS.nb_classes
    print "Batch size:    %d" % FLAGS.batch_size
    print "Hidden layer:  %d" % FLAGS.hidden_dim
    print "Train epochs:  %d" % FLAGS.train_steps
    print "Learning rate: %f" % FLAGS.lr

    print "#" * 67
    print "Training process start."
    print "#" * 67

    # if FLAGS.model == 'LSTM':
    #     Model_type = tagger.LSTM_NER
    # elif FLAGS.model == 'BLSTM':
    #     Model_type = tagger.Bi_LSTM_NER
    # elif FLAGS.model == 'CNNBLSTM':
    #     Model_type = tagger.CNN_Bi_LSTM_NER
    # else:
    #     raise TypeError("Unknow model type % " % FLAGS.model)

    model = Hybrid_LSTM_tagger(
        nb_words, FLAGS.emb_dim, emb_mat, FLAGS.feat_size, FLAGS.hidden_dim,
        FLAGS.nb_classes, FLAGS.max_len, FLAGS.fine_tuning, FLAGS.dropout,
        FLAGS.batch_size, len(template.template), FLAGS.window, FLAGS.l2_reg)

    pred_test, test_loss, test_acc = model.run(
        train_X, train_F, train_Y, train_lens,
        valid_X, valid_F, valid_Y, valid_lens,
        test_X, test_F, test_Y, test_lens,
        FLAGS)

    print "Test loss: %f, accuracy: %f" % (test_loss, test_acc)
    # pred_test = [pred_test[i][:test_lens[i]] for i in xrange(len(pred_test))]
    pred_test_label = convert_id_to_word(pred_test, idx2label)
    if FLAGS.eval_test:
        res_test, pred_test_label = evaluate(pred_test_label, test_labels)
        print "Test F1: %f, P: %f, R: %f" % (res_test['f1'], res_test['p'], res_test['r'])
    original_text = [[item['w'] for item in sent] for sent in test_corpus]
    write_prediction(FLAGS.output_dir + 'prediction.utf8',
                     original_text, pred_test_label)

    print "Saving feature dicts..."
    save_dicts(FLAGS.output_dir, FLAGS.feats2idx,
               FLAGS.words2idx, FLAGS.label2idx)

if __name__ == "__main__":
    tf.app.run()
