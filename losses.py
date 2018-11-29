import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributions as D

from nn_helpers.utils import type_tint, type_tdouble, type_tfloat


def loss_bce_kld(x, x_hat, mu, log_var):
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    BCE = F.binary_cross_entropy(x_hat.view(-1, 1), x.view(-1, 1), reduction='elementwise_mean')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return KLD + BCE


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.n_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.n_bad_epochs = 0
            self.best = metrics
        else:
            self.n_bad_epochs += 1

        if self.n_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode' + mode + ' unknown')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
