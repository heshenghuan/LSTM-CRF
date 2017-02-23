#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-22 21:05:01

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan

Some constants may need to use.
"""


OOV = '_OOV_'
GOLD_TAG = 'GoldNER'
POS = 'POS'
SEG = 'Segmentation'
PRED_TAG = 'NER'
task = 'ner'
MAX_LEN = 140  # Max sequence length, because Weibo's input limitation

# Feature templates
local_templates = (
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),
    (('w',  1), ),
    (('w',  2), ),
    (('w', -2), ('w',  -1)),
    (('w', -1), ('w',  0)),
    (('w',  0), ('w',  1)),
    (('w',  1), ('w',  2)),
    (('w',  -1), ('w',  1)),
)

LogP_ZERO = float('-inf')
LogP_INF = float('inf')
LogP_ONE = 0.0
