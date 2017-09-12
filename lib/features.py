#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import codecs as cs
from parameters import START, END


class Template(object):
    """
    Context-based feature template.
    """
    def __init__(self, fn=None, sep=', ', suffix=True):
        """
        Returns a templates instance from file.

        fn: templates file path
        sep: separator
        suffix: Enable or disable feature suffix which used to identify
                different templates
        """
        self._fields = None
        self._template = None
        self._suffix = None
        self._valid = False

        if fn is None:
            raise ValueError('Initializer got a \'None\' instead a file path.')
        else:
            self.build(fn, sep, suffix)

    def build(self, fn, sep, suffix):
        self._fields = []
        self._template = []
        with cs.open(fn, 'r', 'utf8') as src:
            lines = src.read().strip().split('\n')
            # read fields from file
            self._fields = lines[1].strip().split()
            # read templates
            for i in range(3, len(lines)):
                temp = []
                line = lines[i]
                for t in line.strip().split(sep):
                    field, offset = t.strip().split(':')
                    if field not in self._fields:
                        p = os.path.abspath(fn)
                        e = "Unknow field %s, File \"%s\", line %d" % (field, p, i+1)
                        raise ValueError(e)
                    temp.append((field, int(offset)))
                self._template.append(temp)
            self._suffix = suffix
            self._valid = True

    def __str__(self):
        string = str(self.__class__) + '\n'
        string += "Fields(%s)\n" % (', '.join("'%s'" % f for f in self.fields))
        string += '' if self._suffix else "Suffix disabled\n" + 'Templates:\n'
        string += ",\n".join(str(t) for t in self._template)
        return string

    @property
    def template(self):
        return self._template

    @template.setter
    def template(self, *args, **kwargs):
        pass

    @property
    def fields(self):
        return self._fields

    @fields.setter
    def fields(self, *args, **kwargs):
        pass

    @property
    def valid(self):
        return self._valid

    @property
    def suffix(self):
        return self._suffix


# Field names of the input data.
# fields = ['w', 'y']

# Attribute templates.
# templates = (
#     (('w', -2), ),
#     (('w', -1), ),
#     (('w',  0), ),
#     (('w',  1), ),
#     (('w',  2), ),
#     (('w', -2), ('w', -1)),
#     (('w', -1), ('w',  0)),
#     (('w',  0), ('w',  1)),
#     (('w',  1), ('w',  2)),
#     (('w', -1), ('w',  1)),
#     )


def readiter(data, names):
    """
    Return an iterator for item sequences read from a file object.
    This function reads a sequence from a file object L{fi}, and
    yields the sequence as a list of mapping objects. Each line
    (item) from the file object is split by the separator character
    L{sep}. Separated values of the item are named by L{names},
    and stored in a mapping object. Every item has a field 'F' that
    is reserved for storing features.

    @type   data:     array
    @param  data:     The data array.
    @type   names:  tuple
    @param  names:  The list of field names.
    @rtype          list of mapping objects
    @return         An iterator for sequences.
    """
    X = []
    for line in data:
        if len(line) != len(names):
            raise ValueError('Too many/few fields (%d) for %r\n' % (len(line), names))
        item = {'F': []}    # 'F' is reserved for features.
        for i in range(len(names)):
            item[names[i]] = line[i]
        X.append(item)
    return X


def apply_templates(X, templates=None, suffix=True):
    """
    Generate features for an item sequence by applying feature templates.
    A feature template consists of a tuple of (name, offset) pairs,
    where name and offset specify a field name and offset from which
    the template extracts a feature value. Generated features are stored
    in the 'F' field of each item in the sequence.

    @type   X:  A list of dict{'w':w, 'r':r, 'y':y, F:[]}
    @param  X:  The item sequence.
    @type   template:   list
    @param  template:   The feature template.
    """
    # print 'in apply templates! input:', X
    if templates is None:
        raise TypeError("templates should be iterable type not NoneType")
    length = len(X)
    for template in templates:
        name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
        for t in range(len(X)):
            values = []
            for field, offset in template:
                p = t + offset
                if p < 0:
                    values.append(START)
                elif p >= length:
                    values.append(END)
                else:
                    values.append(X[p][field])
            if suffix:
                X[t]['F'].append('%s=%s' % (name, '|'.join(values)))
            else:
                X[t]['F'].append('%s' % ('|'.join(values)))
    return X


def escape(src):
    """
    Escape colon characters from feature names.

    @type   src:    str
    @param  src:    A feature name
    @rtype          str
    @return         The feature name escaped.
    """
    return src.replace(':', '__COLON__')


def output_features(fo, X, field=''):
    """
    Output features (and reference labels) of a sequence in CRFSuite
    format. For each item in the sequence, this function writes a
    reference label (if L{field} is a non-empty string) and features.

    @type   fo:     file
    @param  fo:     The file object.
    @type   X:      list of mapping objects
    @param  X:      The sequence.
    @type   field:  str
    @param  field:  The field name of reference labels.
    """
    for t in range(len(X)):
        if field:
            fo.write('%s' % X[t][field])
        for a in X[t]['F']:
            if isinstance(a, str):
                fo.write('\t%s' % escape(a))
            else:
                fo.write('\t%s:%f' % (escape(a[0]), a[1]))
        fo.write('\n')
    fo.write('\n')


def feature_extractor(X, templates=None):
    # Apply attribute templates to obtain features (in fact, attributes)
    if templates is None:
        raise TypeError("templates should be iterable type not NoneType")
    return apply_templates(X, templates.template, templates.suffix)
