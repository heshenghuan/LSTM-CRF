#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017-02-20 17:49:19

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""


def eval_ner(pred, gold):
    """
    Utilities for evaluation.
    2017-02-15 19:29:49 还未阅读这部分
    """
    print 'Evaluating...'
    eval_dict = {}    # value=[#match, #pred, #gold]
    for p_1sent, g_1sent in zip(pred, gold):
        in_correct_chunk = False
        last_pair = ['^', '$']
        for p, g in zip(p_1sent, g_1sent):
            tp = p.split('-')
            tg = g.split('-')
            if len(tp) == 2:
                if tp[1] not in eval_dict:
                    eval_dict[tp[1]] = [0] * 3
                if tp[0] == 'B' or tp[0] == 'S':
                    eval_dict[tp[1]][1] += 1
            if len(tg) == 2:
                if tg[1] not in eval_dict:
                    eval_dict[tg[1]] = [0] * 3
                if tg[0] == 'B' or tg[0] == 'S':
                    eval_dict[tg[1]][2] += 1

            if p != g or len(tp) == 1:
                if in_correct_chunk and tp[0] != 'I' and tg[0] != 'I' and tp[0] != 'E' and tg[0] != 'E':
                    assert last_pair[0] == last_pair[1]
                    eval_dict[last_pair[0]][0] += 1
                in_correct_chunk = False
                last_pair = ['^', '$']
            else:
                if tg[0] == 'B' or tg[0] == 'S':
                    if in_correct_chunk:
                        assert (last_pair[0] == last_pair[1])
                        eval_dict[last_pair[0]][0] += 1
                    last_pair = [tp[-1], tg[-1]]
                if tg[0] == 'B':
                    in_correct_chunk = True
                if tg[0] == 'S':
                    eval_dict[last_pair[0]][0] += 1
                    in_correct_chunk = False
        if in_correct_chunk:
            assert last_pair[0] == last_pair[1]
            eval_dict[last_pair[0]][0] += 1
    agg_measure = [0.0] * 3
    agg_counts = [0] * 3
    for k, v in eval_dict.items():
        agg_counts = [sum(x) for x in zip(agg_counts, v)]
        prec = float(v[0]) / v[1] if v[1] != 0 else 0.0
        recall = float(v[0]) / v[2] if v[2] != 0 else 0.0
        F1 = 2 * prec * recall / \
            (prec + recall) if prec != 0 and recall != 0 else 0.0
        agg_measure[0] += prec
        agg_measure[1] += recall
        agg_measure[2] += F1
        print k + ':', v[0], '\t', v[1], '\t', v[2], '\t', prec, '\t', recall, '\t', F1
    agg_measure = [v / len(eval_dict) for v in agg_measure]
    print 'Macro average:', '\t', agg_measure[0], '\t', agg_measure[1], '\t', agg_measure[2]
    prec = float(agg_counts[0]) / agg_counts[1] if agg_counts[1] != 0 else 0.0
    recall = float(agg_counts[0]) / \
        agg_counts[2] if agg_counts[2] != 0 else 0.0
    F1 = 2 * prec * recall / \
        (prec + recall) if prec != 0 and recall != 0 else 0.0
    print 'Micro average:', agg_counts[0], '\t', agg_counts[1], '\t', agg_counts[2], '\t', prec, '\t', recall, '\t', F1
    return {'p': prec, 'r': recall, 'f1': F1}


def error_analysis(words, preds, golds, idx_to_word):
    print 'error analysis!!!'
    for w_1sent, p_1sent, g_1sent in zip(words, preds, golds):
        for w, p, g in zip(w_1sent, p_1sent, g_1sent):
            if p != g:
                print idx_to_word[w], p, g
    print 'end of error analysis!!!!'
