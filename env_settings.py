#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os

# BASE_DIR = r'/Users/heshenghuan/Projects/linear_chainCRF/'
BASE_DIR = os.getcwd() + '/'
MODEL_DIR = BASE_DIR + r'models/'
DATA_DIR = BASE_DIR + r'data/'
EMB_DIR = BASE_DIR + r'embeddings/'
OUTPUT_DIR = BASE_DIR + r'export/'
LOG_DIR = BASE_DIR + r'Summary/'

if __name__ == "__main__":
    # These three dirs must be exist, because they are default output dirs
    dirs = [MODEL_DIR, OUTPUT_DIR, LOG_DIR]
    for d in dirs:
        if not os.path.exists(d):
            print d, "not exists."
            print "Automatically created it."
            os.makedirs(d)
