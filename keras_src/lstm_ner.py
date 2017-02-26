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
import cPickle as pickle
from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding
from keras.optimizers import RMSprop
from constant import MAX_LEN, MODEL_DIR


class lstm_ner():

    def __init__(self):
        """
        Returns an instance of lstm_ner, but not initialized.
        """
        self.model = None
        self.embedding_layer = None
        self.lstm_layer = None
        self.optimizer = None

    def initialization(self, nb_words, emb_dim, emb_matrix, output_dim,
                       lr=0.01, batch_size=None, time_steps=MAX_LEN,
                       fine_tuning=False):
        """
        Initialize an instance.
        """
        self.model = Sequential()
        self.embedding_layer = Embedding(nb_words + 1,
                                         emb_dim,
                                         weights=[emb_matrix],
                                         input_length=time_steps,
                                         mask_zero=True, trainable=fine_tuning)
        self.lstm_layer = LSTM(output_dim=output_dim,
                               return_sequences=True,
                               input_shape=(batch_size, time_steps, emb_dim))

        self.model.add(self.embedding_layer)
        self.model.add(self.lstm_layer)
        self.optimizer = RMSprop(lr=lr)

    def summary(self):
        """Print summary information of this lstm_ner to screen."""
        assert self.model is not None, (
            "You must call the initialization function before summay."
        )
        self.model.summary()

    def predict(self, x=None, seq_len=[], verbose=0):
        assert self.model is not None, (
            "You must call the initialization function before predict."
        )
        assert isinstance(x, np.ndarray)
        self.set_seq_length(seq_len)
        ans = self.model.predict(x, verbose=verbose)
        self.remove_seq_length()
        return ans

    def set_seq_length(self, seq_len=[]):
        """
        """
        assert self.model is not None, (
            "You must call the initialization function before set_seq_length."
        )
        assert len(seq_len) != 0, (
            "You must pass a list that do stored each sequences' length."
        )
        self.lstm_layer.input_length = seq_len

    def remove_seq_length(self):
        assert self.model is not None, (
            "You must call the initialization function before remove_seq_length."
        )
        self.lstm_layer.input_length = None

    def compile(self, optimizer=None, loss='categorical_crossentropy',
                metrics=['accuracy'], **kwargs):
        assert self.model is not None, (
            "You must call the initialization function before compile."
        )
        if optimizer is None:
            optimizer = self.optimizer
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                           **kwargs)

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0,
            sequences_length=None, **kwargs):
        """"""
        assert self.model is not None, (
            "You must call the initialization function before fit."
        )
        self.set_seq_length(sequences_length)
        self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks,
                       validation_split, validation_data, shuffle,
                       class_weight, sample_weight, initial_epoch,
                       **kwargs)
        self.remove_seq_length()
        return

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None,
                 **kwargs):
        """"""
        assert self.model is not None, (
            "You must call the initialization function before evaluate."
        )
        return self.model.evaluate(x, y, batch_size, verbose, sample_weight,
                                   **kwargs)

    # def call(self, name):
    #     return getattr(self.model, name)()

    def save(self, filepath=MODEL_DIR+'model.h5', overwrite=True):
        assert self.model is not None, (
            "You must call the initialization function before save."
        )
        return self.model.save(filepath, overwrite)

    def load(self, filepath=MODEL_DIR+'model.h5', custom_objects=None):
        """"""
        self.model = load_model(filepath, custom_objects)
        self.embedding_layer = self.model.layers[0]
        self.lstm_layer = self.model.layers[1]
        self.optimizer = RMSprop()
