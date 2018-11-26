import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributions as D
from utils import t_type_int, t_type_double, t_type_float


def kl_divergence_np(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def bce_accuracy(pred_logits, targets, size_average=True):
    # https://github.com/jramapuram/helpers/blob/master/metrics.py
    cuda = pred_logits.is_cuda
    input_pred = F.sigmoid(pred_logits).type(t_type_int(cuda))
    targets = targets.type(t_type_int(cuda))
    reduction_fn = torch.mean if size_average else torch.sum
    return reduction_fn(input_pred.eq(targets).cpu().type(t_type_float(False)))
