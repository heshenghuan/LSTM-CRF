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
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from evaluate_util import eval_ner
from constant import MAX_LEN


class lstm_ner():
    def __init__(self):
        self.model = None
        self.embedding_layer = None
        self.lstm_layer = None

    def initialization(self, nb_words, emb_dim, emb_matrix, output_dim,
                       batch_size=None, time_steps=MAX_LEN, fine_tuning=False):
        self.model = Sequential()
        self.embedding_layer = Embedding(nb_words + 1,
                                         emb_dim,
                                         weights=[emb_matrix],
                                         input_length=MAX_LEN,
                                         mask_zero=True, trainable=fine_tuning)
        self.lstm_layer = LSTM(output_dim=output_dim,
                               return_sequences=True,
                               input_shape=(batch_size, time_steps, emb_dim))

        self.model.add(self.embedding_layer)
        self.model.add(self.lstm_layer)

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print "You must call the initialization function before summay."

    def predict(self, samps):
        assert isinstance(samps, np.array)
        assert samps.shape[0] > None and samps.shape[1] == MAX_LEN
        return self.model.predict(samps)

    def viterbi(self, lb_probs):
        pass

    def sequence_labeling(self, sntc):
        pass
