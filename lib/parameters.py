#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan

Some env params may need to use.
"""


OOV = '_OOV_'
START = '_S_'
END = '_E_'
GOLD_TAG = 'GoldNER'
PRED_TAG = 'NER'
task = 'ner'
MAX_LEN = 175  # Max sequence length, because Weibo's input limitation

LogP_ZERO = float('-inf')
LogP_INF = float('inf')
LogP_ONE = 0.0
FloatX = 'float32'
