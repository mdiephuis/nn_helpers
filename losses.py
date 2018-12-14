import torch
import numpy as np
import torch.nn.functional as F
from nn_helpers.utils import sample_normal
from nn_helpers.utils import type_tfloat


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


def MSE_kernel(p, q, sigma, use_cuda):
    p_tiled = p.expand(p.size(0), q.size(0), p.size(1))
    q_tiled = q.expand(p.size(0), q.size(0), p.size(1)).transpose(0, 1)

    loss_func = torch.nn.MSELoss(reduction='mean')
    nom = - loss_func(p_tiled, q_tiled)

    const_val = type_tfloat(use_cuda)(1).zero_() + 2.0
    sigma = type_tfloat(use_cuda)(1).zero_() + sigma
    denom = torch.mul(const_val, torch.pow(sigma, 2))

    return torch.exp(torch.div(nom, denom)).item()


def loss_mmd(p, q, kernel_func, sigma, use_cuda):
    pp = kernel_func(p, p, sigma, use_cuda)
    qq = kernel_func(q, q, sigma, use_cuda)
    pq = kernel_func(p, q, sigma, use_cuda)

    return pp + qq - 2 * pq


def loss_infovae(x, x_hat, z, sigma, use_cuda):
    true_samples = sample_normal(z.size(), use_cuda)
    mmd = loss_mmd(true_samples, z, MSE_kernel, sigma, use_cuda)

    nll_func = torch.nn.MSELoss(reduction='mean')
    nll = nll_func(x, x_hat)

    return mmd + nll


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
