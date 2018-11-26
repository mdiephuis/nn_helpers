import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributions as D


def t_type_double(use_cuda=False):
    return torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def t_type_float(use_cuda=False):
    return torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def t_type_int(use_cuda=False):
    return torch.cuda.IntTensor if use_cuda else torch.IntTensor


def t_type_long(use_cuda=False):
    return torch.cuda.LongTensor if use_cuda else torch.LongTensor


def one_hot_np(labels, n_class):
    vec = np.zeros((labels.shape[0], n_class))
    for ind, label in enumerate(labels):
        vec[ind, label] = 1
    return vec


def one_hot(labels, n_class):

    # Ensure labels are [N x 1]
    if len(list(labels.size())) == 1:
        labels = labels.unsqueeze()

    mask = t_type_double()(labels.size(0), n_class).fill_(0)

    # scatter dimension, position indices, fill_value
    mask.scatter_(1, labels, 1)
