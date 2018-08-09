# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np


def compute_recall_ks(probas):
    recall_k = {}
    for group_size in [2, 5, 10]:
        recall_k[group_size] = {}
        print ('group_size: %d' % group_size)
        for k in [1, 2, 5]:
            if k < group_size:
                recall_k[group_size][k] = recall(probas, k, group_size)
                print ('recall@%d' % k, recall_k[group_size][k])
    return recall_k

def recall(probas, k, group_size):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    for i in range(n_batches):
        batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        indices = np.argpartition(batch, -k)[-k:]
        if 0 in indices:
            n_correct += 1
    return float(n_correct) / (len(probas) / test_size)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")